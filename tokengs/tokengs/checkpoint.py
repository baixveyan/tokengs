"""训练 checkpoint 的持久化事务与可核验来源契约。"""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from pathlib import Path

import torch


SCHEMA_VERSION = 1
RETENTION_POLICY = {
    'name': 'inference-best-resumable-last-and-live-readers', 'stable_versions': 2,
    'replacement_versions': 2, 'temporary_versions': 1,
    'best_payload': ['model.safetensors', 'metadata.json'],
    'last_payload': ['model.safetensors', 'optimizer.pth', 'scheduler.pth', 'metadata.json'],
    'deduplicate_best_last': False, 'gc_after_pointer_fsync': True,
    'locking': 'posix-flock', 'filesystem_contract': 'local-posix-rename-fsync',
}
_VERSION_NAME = re.compile(r'(?:best|last)-e[0-9]{6,}-[0-9a-f]{32}\Z')
_GIB = 1024 ** 3
MIN_FREE_RESERVE_BYTES = 15 * _GIB
FORMAL_CONCURRENT_RUNS = 2

# 仅允许输出、日志设置和同 SHA 初始化文件的位置变化；训练参数不在白名单。
RESUME_ALLOWED_FIELDS = frozenset({
    'workspace', 'resume', 'out_dir', 'use_wandb', 'project_name',
    'experiment_name', 'print_freq', 'log_image_freq',
})


def require_finite(value, name):
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f'{name}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f'{name}[{index}]')
    elif isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f'Non-finite {name}')
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise RuntimeError(f'Non-finite {name}: {value}')


def sha256_file(path):
    if path is None or path == 'None':
        return None
    if not os.path.isfile(path):
        raise RuntimeError(f'Checkpoint file is missing: {path}')
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _options_snapshot(opt):
    return json.loads(json.dumps(asdict(opt), allow_nan=False))


def _critical_options(snapshot):
    return {key: value for key, value in snapshot.items() if key not in RESUME_ALLOWED_FIELDS}


def _world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _configuration_sha256(snapshot, world_size):
    return _json_sha256({'options': _critical_options(snapshot), 'world_size': world_size})


def options_sha256(opt):
    return _configuration_sha256(_options_snapshot(opt), _world_size())


def capture_provenance(opt):
    """必须在自动恢复修改 opt.resume 之前捕获当前事实。"""
    snapshot = _options_snapshot(opt)
    provenance = {
        'seed': opt.seed,
        'use_geo_losses': opt.use_geo_losses,
        'world_size': _world_size(),
        'config_sha256': _configuration_sha256(snapshot, _world_size()),
        'initialization_checkpoint_sha256': sha256_file(opt.resume),
        'options': snapshot,
        'options_snapshot_sha256': _json_sha256(snapshot),
        'resume_allowed_fields': sorted(RESUME_ALLOWED_FIELDS),
    }
    opt._checkpoint_provenance = provenance
    return provenance


def validate_provenance(metadata, current):
    for key in ('seed', 'use_geo_losses', 'world_size', 'config_sha256', 'initialization_checkpoint_sha256'):
        if key not in metadata or metadata[key] != current[key]:
            raise RuntimeError(f'Incompatible resume provenance: {key}')
    saved = metadata.get('options')
    if not isinstance(saved, dict):
        raise RuntimeError('Resume checkpoint lacks verifiable options provenance')
    if (metadata.get('options_snapshot_sha256') != _json_sha256(saved)
            or metadata['config_sha256'] != _configuration_sha256(saved, metadata['world_size'])
            or saved.get('seed') != metadata['seed']
            or saved.get('use_geo_losses') != metadata['use_geo_losses']):
        raise RuntimeError('Inconsistent resume options/config SHA provenance')
    if metadata.get('resume_allowed_fields') != sorted(RESUME_ALLOWED_FIELDS):
        raise RuntimeError('Incompatible resume allowed-fields contract')
    changes = []
    for key in sorted(set(saved) | set(current['options'])):
        before, after = saved.get(key), current['options'].get(key)
        if before != after:
            if key not in RESUME_ALLOWED_FIELDS:
                raise RuntimeError(f'Incompatible resume option: {key}')
            changes.append({'field': key, 'before': before, 'after': after})
    return changes


def synchronized_call(accelerator, callback, *, main_only=False):
    """同步可捕获的失败，防止主 rank 异常后其他 rank 留在末尾 barrier。"""
    result, error = None, None
    try:
        if not main_only or accelerator.is_main_process:
            result = callback()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        outcomes = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(outcomes, (error, result))
        failures = [item[0] for item in outcomes if item[0] is not None]
        if failures:
            raise RuntimeError('Checkpoint operation failed: ' + '; '.join(failures))
        return outcomes[0][1] if main_only else result
    if error is not None:
        raise RuntimeError(error)
    return result


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path, metadata):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_save_state(path, state):
    temporary = path.with_name(f'.{path.name}.tmp')
    with open(temporary, 'wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_pointer(root, kind, version):
    temporary = root / f'.{kind}.{uuid.uuid4().hex}.tmp'
    os.symlink(version, temporary)
    os.replace(temporary, root / kind)
    _fsync_directory(root)


def _publish_version(root, kind, accelerator, model, optimizer, scheduler, metadata,
                     *, resumable):
    versions = root / 'versions'
    versions.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root)
    name = f'{kind}-e{metadata["epoch"]:06d}-{uuid.uuid4().hex}'
    temporary = versions / f'.{name}.tmp'
    temporary.mkdir()
    accelerator.save_model(model, str(temporary))
    if not (temporary / 'model.safetensors').is_file():
        raise RuntimeError('Checkpoint requires a complete, unsharded model.safetensors')
    payload = ['model.safetensors']
    if resumable:
        _atomic_save_state(temporary / 'optimizer.pth', optimizer.state_dict())
        _atomic_save_state(temporary / 'scheduler.pth', scheduler.state_dict())
        payload.extend(('optimizer.pth', 'scheduler.pth'))
    checksums = {}
    for filename in payload:
        path = temporary / filename
        with open(path, 'rb') as handle:
            os.fsync(handle.fileno())
        checksums[filename] = sha256_file(path)
    metadata = metadata | {'checkpoint_schema': SCHEMA_VERSION, 'checkpoint_kind': kind,
                           'version': f'versions/{name}', 'file_sha256': checksums,
                           'payload_files': payload + ['metadata.json'], 'resumable': resumable}
    atomic_write_json(temporary / 'metadata.json', metadata)
    _fsync_directory(temporary)
    os.replace(temporary, versions / name)
    _fsync_directory(versions)
    # 唯一提交点：读者只跟随此原子指针，不扫描尚未提交的版本目录。
    _publish_pointer(root, kind, metadata['version'])
    return metadata


def _read_version(root, version, kind):
    relative = Path(version)
    if (relative.is_absolute() or len(relative.parts) != 2
            or relative.parts[0] != 'versions' or relative.parts[1].startswith('.')):
        raise RuntimeError(f'Invalid checkpoint version: {version}')
    directory = root / relative
    if directory.is_symlink() or directory.resolve().parent != (root / 'versions').resolve():
        raise RuntimeError(f'Invalid checkpoint directory: {directory}')
    with open(directory / 'metadata.json', encoding='utf-8') as handle:
        metadata = json.load(handle)
    if metadata.get('best_val_psnr') is not None:
        metadata['best_val_psnr'] = float(metadata['best_val_psnr'])
    require_finite(metadata.get('best_val_psnr'), 'restored best_val_psnr')
    require_finite(metadata.get('val_metrics'), 'restored val_metrics')
    if metadata.get('val_metrics') is not None:
        metadata['val_metrics']['psnr'] = float(metadata['val_metrics']['psnr'])
        require_finite(metadata['val_metrics']['psnr'], 'restored validation PSNR')
    stored_kind = metadata.get('checkpoint_kind')
    checksums = metadata.get('file_sha256', {})
    legacy_best_alias = (kind == 'last' and stored_kind == 'best'
                         and all(checksums.get(name) for name in
                                 ('model.safetensors', 'optimizer.pth', 'scheduler.pth')))
    if (metadata.get('checkpoint_schema') != SCHEMA_VERSION
            or not (stored_kind == kind or legacy_best_alias)
            or metadata.get('version') != version):
        raise RuntimeError('Invalid checkpoint schema/kind/version')
    required = ('model.safetensors', 'optimizer.pth', 'scheduler.pth') if kind == 'last' else ('model.safetensors',)
    for filename in required:
        if not checksums.get(filename) or sha256_file(directory / filename) != checksums[filename]:
            raise RuntimeError(f'Checkpoint checksum mismatch: {filename}')
    if kind == 'last' and metadata.get('resumable', True) is not True:
        raise RuntimeError('Last checkpoint is not resumable')
    if stored_kind == 'best':
        if (metadata.get('metric_split') != 'val' or metadata.get('val_metrics') is None
                or metadata.get('best_val_psnr') is None
                or metadata['best_val_psnr'] != metadata['val_metrics']['psnr']):
            raise RuntimeError('Invalid best validation metric')
    if legacy_best_alias:
        # 兼容旧事务中由完整 best 版本兼任 last 的已提交 checkpoint。
        metadata = metadata | {'best_checkpoint': _best_reference(root, metadata)}
    return metadata


def _read_committed(root, kind):
    pointer = root / kind
    if not os.path.lexists(pointer):
        return None
    if not pointer.is_symlink():
        raise RuntimeError('Legacy checkpoint cannot be resumed without transactional provenance')
    return _read_version(root, os.readlink(pointer), kind)


def _best_reference(root, metadata):
    if metadata is None:
        return None
    return {
        'version': metadata['version'],
        'path': f'{metadata["version"]}/model.safetensors',
        'epoch': metadata['epoch'],
        'metric': metadata['best_val_psnr'],
        'model_sha256': metadata['file_sha256']['model.safetensors'],
        'metadata_sha256': sha256_file(root / metadata['version'] / 'metadata.json'),
    }


def _validate_best_reference(root, reference, current):
    if reference is None:
        return None
    metadata = _read_version(root, reference['version'], 'best')
    validate_provenance(metadata, current)
    if reference != _best_reference(root, metadata):
        raise RuntimeError('Checkpoint best reference checksum/epoch/metric mismatch')
    return metadata


@contextmanager
def checkpoint_lock(root, *, exclusive=False):
    """目录级跨进程锁：串行化写事务，读者在此锁内取得版本租约。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / '.transaction.lock', 'a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _VersionLease:
    def __init__(self, directory):
        self.root = directory.parent.parent
        self.handle = open(directory / '.read.lock', 'a+b')
        fcntl.flock(self.handle, fcntl.LOCK_SH)

    def close(self):
        if not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

    def __del__(self):
        if hasattr(self, 'handle'):
            self.close()


def release_checkpoint_read_lock(opt):
    """model、optimizer、scheduler 全部载入后才允许回收所选恢复版本。"""
    leases = getattr(opt, '_checkpoint_read_leases', [])
    roots = {lease.root for lease in leases}
    for lease in leases:
        lease.close()
    opt._checkpoint_read_leases = []
    for root in roots:
        recover_checkpoint_storage(root)


def _version_directory(root, version):
    if (root / 'versions').is_symlink():
        raise RuntimeError('Checkpoint versions must be a local managed directory')
    relative = Path(version)
    if (relative.is_absolute() or len(relative.parts) != 2
            or relative.parts[0] != 'versions' or not _VERSION_NAME.fullmatch(relative.name)):
        raise RuntimeError(f'Invalid checkpoint version: {version}')
    directory = root / relative
    if directory.is_symlink() or directory.resolve().parent != (root / 'versions').resolve():
        raise RuntimeError(f'Invalid checkpoint directory: {directory}')
    return directory


@contextmanager
def read_checkpoint(root, kind='last', *, version=None):
    """统一读取入口；指针或显式 UUID 版本必须在上下文中完成全部读取。"""
    root = Path(root)
    with ExitStack() as stack:
        with checkpoint_lock(root):
            if version is None:
                pointer = root / kind
                if not os.path.lexists(pointer):
                    raise RuntimeError(f'No committed checkpoint: {kind}')
                if not pointer.is_symlink():
                    raise RuntimeError('Legacy checkpoint cannot be resumed without transactional provenance')
                version = os.readlink(pointer)
            directory = _version_directory(root, version)
            lease = _VersionLease(directory)
            # ExitStack 后进先出：全部版本租约释放后立即回收，避免稳定时刻遗留旧版本。
            stack.callback(recover_checkpoint_storage, root)
            stack.callback(lease.close)
            # 目录锁内选定并取得租约，metadata 和 SHA 校验也由同一版本租约保护。
            metadata = _read_version(root, version, kind)
            reference = metadata.get('best_checkpoint')
            if reference is not None and reference['version'] != version:
                lease = _VersionLease(_version_directory(root, reference['version']))
                stack.callback(lease.close)
        yield directory, metadata


@contextmanager
def checkpoint_read_path(resume):
    """托管指针与已解析 UUID 共用读取上下文；官方裸文件保持原行为。"""
    path = Path(resume).absolute()
    if (path.name == 'model.safetensors' and path.parent.name in ('best', 'last')
            and path.parent.is_symlink()):
        with read_checkpoint(path.parent.parent, path.parent.name) as (directory, metadata):
            yield directory / path.name, metadata
        return
    if (path.name == 'model.safetensors' and path.parent.parent.name == 'versions'
            and _VERSION_NAME.fullmatch(path.parent.name)):
        root = path.parent.parent.parent
        kind = path.parent.name.split('-', 1)[0]
        with read_checkpoint(root, kind, version=f'versions/{path.parent.name}') as (directory, metadata):
            yield directory / path.name, metadata
        return
    yield path, None


def load_checkpoint_model(path):
    """加载模型的安全裸路径适配器；官方初始化不进入本 run 的 GC 管理域。"""
    from safetensors.torch import load_file

    path = Path(path).absolute()
    if (path.name == 'model.safetensors' and path.parent.name in ('best', 'last')
            and path.parent.is_symlink()):
        with read_checkpoint(path.parent.parent, path.parent.name) as (directory, _):
            return load_file(str(directory / path.name), device='cpu')
    if (path.name == 'model.safetensors' and path.parent.parent.name == 'versions'
            and _VERSION_NAME.fullmatch(path.parent.name)):
        root = path.parent.parent.parent
        with ExitStack() as stack:
            with checkpoint_lock(root):
                directory = _version_directory(root, f'versions/{path.parent.name}')
                stack.callback(recover_checkpoint_storage, root)
                lease = _VersionLease(directory)
                stack.callback(lease.close)
            return load_file(str(directory / path.name), device='cpu')
    if path.suffix == '.safetensors':
        return load_file(str(path), device='cpu')
    return torch.load(str(path), map_location='cpu')


def _retained_versions(root):
    """只解析指针和 last 的 best 引用；有损坏时 fail-closed，不执行删除。"""
    retained = set()
    for kind in ('best', 'last'):
        pointer = root / kind
        if not os.path.lexists(pointer):
            continue
        if not pointer.is_symlink():
            raise RuntimeError('Legacy checkpoint cannot be resumed without transactional provenance')
        version = os.readlink(pointer)
        directory = _version_directory(root, version)
        with open(directory / 'metadata.json', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('version') != version or metadata.get('checkpoint_schema') != SCHEMA_VERSION:
            raise RuntimeError('Invalid checkpoint schema/version during GC')
        retained.add(version)
        reference = metadata.get('best_checkpoint') if kind == 'last' else None
        if reference is not None:
            referenced = _version_directory(root, reference['version'])
            if not (referenced / 'metadata.json').is_file():
                raise RuntimeError('Missing best reference during GC')
            retained.add(reference['version'])
    return retained


def _remove_retired(directory):
    shutil.rmtree(directory)


def _gc_locked(root, *, disk_estimate=None):
    retained = _retained_versions(root)
    removed, protected = [], []
    versions = root / 'versions'
    if versions.is_symlink():
        raise RuntimeError('Checkpoint versions must be a local managed directory')
    if versions.exists():
        for directory in sorted(versions.iterdir()):
            name = directory.name
            if directory.is_symlink() or not directory.is_dir():
                continue
            temporary = name.startswith('.') and name.endswith('.tmp') and _VERSION_NAME.fullmatch(name[1:-4])
            retired = name.startswith('.gc-') and _VERSION_NAME.fullmatch(name[4:])
            managed = _VERSION_NAME.fullmatch(name)
            if temporary or retired:
                _remove_retired(directory)
                removed.append(name)
            elif managed and f'versions/{name}' not in retained:
                # 目录锁阻止新读者进入；不阻塞仍在加载的评估或恢复进程。
                with open(directory / '.read.lock', 'a+b') as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        protected.append(f'versions/{name}')
                        continue
                    retired_path = versions / f'.gc-{name}'
                    os.replace(directory, retired_path)
                    _fsync_directory(versions)
                    _remove_retired(retired_path)
                    removed.append(name)
        _fsync_directory(versions)
    for temporary in root.iterdir():
        if re.fullmatch(r'\.(?:best|last|retention_audit\.json)\.[0-9a-f]{32}\.tmp', temporary.name):
            if temporary.is_symlink() or temporary.is_file():
                temporary.unlink()
    report = {'retention_policy': dict(RETENTION_POLICY), 'retained_versions': sorted(retained),
              'reader_protected_versions': protected, 'removed_versions': removed}
    audit_path = root / 'retention_audit.json'
    if disk_estimate is None and audit_path.is_file():
        with open(audit_path, encoding='utf-8') as handle:
            disk_estimate = json.load(handle).get('disk_peak_estimate')
    report['disk_peak_estimate'] = disk_estimate
    atomic_write_json(audit_path, report)
    return report


def recover_checkpoint_storage(root):
    """启动清理与提交后 GC 共用幂等逻辑，只涉及本 run 的 UUID 版本。"""
    root = Path(root)
    with checkpoint_lock(root, exclusive=True):
        return _gc_locked(root)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def checkpoint_component_bytes(model, optimizer=None, scheduler=None):
    """包含 buffer；AdamW 尚未创建的两个动量状态也计入，避免启动低估。"""
    state = model.state_dict()
    optimizer_bytes = _tensor_bytes(optimizer.state_dict()) if optimizer is not None else 0
    raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
    if isinstance(raw_optimizer, torch.optim.AdamW):
        params = [p for group in raw_optimizer.param_groups for p in group['params']]
        lazy_bound = sum(2 * p.numel() * p.element_size() + 8 for p in params)
        if any(group.get('amsgrad', False) for group in raw_optimizer.param_groups):
            lazy_bound += sum(p.numel() * p.element_size() for p in params)
        optimizer_bytes = max(optimizer_bytes, lazy_bound)
    return {
        'model_bytes': _tensor_bytes(state) + len(state) * 1024 + 1024 ** 2,
        'optimizer_bytes': optimizer_bytes + 1024 ** 2,
        'scheduler_bytes': (_tensor_bytes(scheduler.state_dict()) if scheduler is not None else 0) + 1024 ** 2,
        'metadata_bytes': 1024 ** 2,
    }


def estimate_checkpoint_disk_peak(*, model_bytes, optimizer_bytes, scheduler_bytes=1024 ** 2,
                                  metadata_bytes=1024 ** 2,
                                  concurrent_runs=FORMAL_CONCURRENT_RUNS,
                                  media_bytes_per_run=0, safety_factor=1.2,
                                  reserve_bytes=MIN_FREE_RESERVE_BYTES,
                                  reader_versions_per_run=0):
    """保守峰值：新旧 best 模型各一份，新旧 last 完整状态各一份。

    best 先提交、last 后提交以保持引用可核验；GC 前会短暂同时存在两代。
    当前存量由实时可用空间自然扣除，额外始终保留至少 15 GiB。
    """
    values = (model_bytes, optimizer_bytes, scheduler_bytes, metadata_bytes,
              media_bytes_per_run, reserve_bytes, reader_versions_per_run)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError('Disk estimate byte counts must be nonnegative integers')
    if (not isinstance(concurrent_runs, int) or concurrent_runs < 1
            or not math.isfinite(safety_factor) or safety_factor < 1):
        raise ValueError('Invalid disk estimate concurrency/safety factor')
    best_version_bytes = model_bytes + metadata_bytes
    last_version_bytes = model_bytes + optimizer_bytes + scheduler_bytes + metadata_bytes
    slots = 4 + reader_versions_per_run
    peak_per_run = 2 * best_version_bytes + 2 * last_version_bytes
    peak_per_run += reader_versions_per_run * last_version_bytes
    checkpoint_bytes = concurrent_runs * peak_per_run
    media_bytes = concurrent_runs * media_bytes_per_run
    return {
        'model_bytes': model_bytes, 'optimizer_bytes': optimizer_bytes,
        'scheduler_bytes': scheduler_bytes, 'metadata_bytes': metadata_bytes,
        'version_bytes': last_version_bytes, 'best_version_bytes': best_version_bytes,
        'last_version_bytes': last_version_bytes,
        'stable_checkpoint_bytes_per_run': best_version_bytes + last_version_bytes,
        'peak_checkpoint_bytes_per_run': peak_per_run, 'stable_versions_per_run': 2,
        'replacement_versions_per_run': 2, 'temporary_versions_per_run': 1,
        'reader_versions_per_run': reader_versions_per_run, 'peak_versions_per_run': slots,
        'concurrent_runs': concurrent_runs, 'checkpoint_peak_bytes': checkpoint_bytes,
        'media_bytes_per_run': media_bytes_per_run, 'media_peak_bytes': media_bytes,
        'safety_factor': safety_factor, 'reserve_bytes': reserve_bytes,
        'required_free_bytes': math.ceil((checkpoint_bytes + media_bytes) * safety_factor) + reserve_bytes,
        'retention_policy': dict(RETENTION_POLICY),
    }


def require_checkpoint_disk_space(path, **estimate_kwargs):
    """正式启动入口可直接调用；使用 statvfs 的非 root 可用空间，空间不足拒绝。"""
    estimate = estimate_checkpoint_disk_peak(**estimate_kwargs)
    path = Path(path)
    while not path.exists():
        path = path.parent
    estimate['available_bytes'] = shutil.disk_usage(path).free
    if estimate['available_bytes'] < estimate['required_free_bytes']:
        raise RuntimeError(f'Insufficient checkpoint disk space: required '
                           f'{estimate["required_free_bytes"]} bytes, available '
                           f'{estimate["available_bytes"]} bytes')
    return estimate


def checkpoint_disk_preflight(opt, accelerator, model, optimizer, scheduler):
    def check():
        components = checkpoint_component_bytes(accelerator.unwrap_model(model), optimizer, scheduler)
        conc = int(os.environ.get("TOKENGS_CONCURRENT_RUNS", FORMAL_CONCURRENT_RUNS))
        estimate = require_checkpoint_disk_space(opt.workspace, concurrent_runs=conc, **components)
        root = Path(opt.workspace) / 'checkpoints'
        with checkpoint_lock(root, exclusive=True):
            _gc_locked(root, disk_estimate=estimate)
        return estimate
    opt._checkpoint_disk_estimate = synchronized_call(accelerator, check, main_only=True)
    return opt._checkpoint_disk_estimate


def restore_checkpoint(opt, accelerator):
    # 不信任 Options.evolve/deepcopy 可能携带的旧缓存，重新读取当前初始化来源。
    release_checkpoint_read_lock(opt)
    current = synchronized_call(accelerator, lambda: capture_provenance(opt))
    root = Path(opt.workspace) / 'checkpoints'
    synchronized_call(accelerator, lambda: recover_checkpoint_storage(root) if root.exists() else None,
                      main_only=True)

    def restore():
        root = Path(opt.workspace) / 'checkpoints'
        last = _read_committed(root, 'last')
        best = _read_committed(root, 'best')
        if last is None and best is None:
            if (Path(opt.workspace) / 'metadata.json').exists():
                raise RuntimeError('Legacy checkpoint lacks transactional resume provenance')
            return {'epoch_start': 0, 'wandb_run_id': None, 'best_val_psnr': None,
                    'best_checkpoint': None}
        if best is not None:
            validate_provenance(best, current)
        if last is None:
            # best 是推理专用载荷；首次提交在 best 与 last 之间崩溃时从官方初始化重跑。
            return {'epoch_start': 0, 'wandb_run_id': None,
                    'best_val_psnr': best['best_val_psnr'] if best else None,
                    'best_checkpoint': _best_reference(root, best),
                    'config_sha256': current['config_sha256'],
                    'initialization_checkpoint_sha256': current['initialization_checkpoint_sha256']}
        changes = validate_provenance(last, current)
        referenced = _validate_best_reference(root, last.get('best_checkpoint'), current)
        if (referenced is None) != (last.get('best_val_psnr') is None):
            raise RuntimeError('Last checkpoint has an unverifiable best metric')
        if referenced and referenced['best_val_psnr'] != last['best_val_psnr']:
            raise RuntimeError('Last checkpoint best metric/reference mismatch')
        candidates = [item for item in (referenced, best) if item is not None]
        recovered_best = max(candidates, key=lambda item: item['best_val_psnr']) if candidates else None
        state_dir = str(root / last['version'])
        protected = {last['version']}
        if recovered_best is not None:
            protected.add(recovered_best['version'])
        for version in sorted(protected):
            opt._checkpoint_read_leases.append(_VersionLease(_version_directory(root, version)))
        return {
            'epoch_start': last['epoch'] + 1,
            'wandb_run_id': last.get('wandb_run_id'),
            'best_val_psnr': recovered_best['best_val_psnr'] if recovered_best else None,
            'best_checkpoint': _best_reference(root, recovered_best),
            'config_sha256': current['config_sha256'],
            'initialization_checkpoint_sha256': current['initialization_checkpoint_sha256'],
            'resume_allowed_changes': last.get('resume_allowed_changes', []) + changes,
            'state_dir': state_dir,
            'model_path': str(Path(state_dir) / 'model.safetensors'),
        }

    def locked_restore():
        # 即使首次启动尚无 root，也必须进入同一锁协议，避免首个提交并发创建指针。
        with checkpoint_lock(root):
            return restore()

    try:
        state = synchronized_call(accelerator, locked_restore)
    except BaseException:
        release_checkpoint_read_lock(opt)
        raise
    if state.get('model_path'):
        opt.resume = state['model_path']
        opt._resume_state_dir = state['state_dir']
        opt._resume_allowed_changes = state['resume_allowed_changes']
        if accelerator.is_main_process:
            print(f'Resuming from {opt.resume}')
    return state


def commit_checkpoint(opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id,
                      val_metrics, best_val_psnr, config_sha256,
                      initialization_checkpoint_sha256, metric_split='val'):
    def validate():
        require_finite(val_metrics, 'checkpoint metrics')
        require_finite(best_val_psnr, 'checkpoint best_val_psnr')
        if metric_split not in ('val', 'test'):
            raise RuntimeError(f'Invalid checkpoint metric split: {metric_split}')
        current = getattr(opt, '_checkpoint_provenance', None) or capture_provenance(opt)
        if (config_sha256 != current['config_sha256']
                or initialization_checkpoint_sha256 != current['initialization_checkpoint_sha256']
                or options_sha256(opt) != current['config_sha256']):
            raise RuntimeError('Checkpoint save provenance does not match current facts')
        val_psnr = None if val_metrics is None else float(val_metrics['psnr'])
        best_psnr = None if best_val_psnr is None else float(best_val_psnr)
        require_finite(val_psnr, 'checkpoint validation PSNR')
        require_finite(best_psnr, 'checkpoint best_val_psnr')
        return current, val_psnr, best_psnr

    current, val_psnr, best_val_psnr = synchronized_call(accelerator, validate)
    accelerator.wait_for_everyone()

    def commit_locked(root):
        require_finite(accelerator.unwrap_model(model).state_dict(), 'checkpoint model')
        require_finite(optimizer.state_dict(), 'checkpoint optimizer')
        require_finite(scheduler.state_dict(), 'checkpoint scheduler')
        _fsync_directory(root.parent)
        recovery = _gc_locked(root)
        # 每次提交也门禁；额外读取租约计入峰值，防止评估长期占用旧版本。
        components = checkpoint_component_bytes(accelerator.unwrap_model(model), optimizer, scheduler)
        runtime = require_checkpoint_disk_space(
            root, **components, concurrent_runs=1, media_bytes_per_run=0,
            reader_versions_per_run=len(recovery['reader_protected_versions']),
        )
        # 启动门禁核查整个队列；提交门禁核查本次增量，不重复预留其他 run 已占用的空间。
        estimate = estimate_checkpoint_disk_peak(
            **components, reader_versions_per_run=len(recovery['reader_protected_versions']))
        estimate['available_bytes'] = runtime['available_bytes']
        estimate['runtime_required_free_bytes'] = runtime['required_free_bytes']
        opt._checkpoint_disk_estimate = estimate
        committed_best = _read_committed(root, 'best')
        if committed_best:
            validate_provenance(committed_best, current)
        reference = _best_reference(root, committed_best)
        if reference is None and best_val_psnr is not None:
            raise RuntimeError('Cannot save a best metric without committed best weights')
        if reference is not None and best_val_psnr != reference['metric']:
            raise RuntimeError('In-memory best metric differs from committed best')
        improved = (metric_split == 'val' and val_psnr is not None
                    and (best_val_psnr is None or val_psnr > best_val_psnr))
        next_best = val_psnr if improved else best_val_psnr
        metadata = current | {
            'epoch': epoch,
            'val_metrics': val_metrics | {'psnr': val_psnr} if metric_split == 'val' and val_metrics is not None else None,
            'best_val_psnr': next_best, 'metric_split': metric_split,
            'resume_allowed_changes': getattr(opt, '_resume_allowed_changes', []),
            'retention_policy': dict(RETENTION_POLICY), 'disk_peak_estimate': estimate,
            'best_reference_mode': 'self' if improved else 'explicit',
        }
        if wandb_run_id:
            metadata['wandb_run_id'] = wandb_run_id
        if improved:
            committed_best = _publish_version(
                root, 'best', accelerator, model, optimizer, scheduler, metadata, resumable=False)
            reference = _best_reference(root, committed_best)
        _publish_version(root, 'last', accelerator, model, optimizer, scheduler,
                         metadata | {'best_checkpoint': reference,
                                     'best_reference_mode': 'explicit'}, resumable=True)
        # 两个新指针均已 rename + fsync。崩溃停在中间时旧 last 引用仍保护旧 best。
        _gc_locked(root, disk_estimate=estimate)
        return next_best

    def commit():
        root = Path(opt.workspace) / 'checkpoints'
        with checkpoint_lock(root, exclusive=True):
            return commit_locked(root)

    next_best = synchronized_call(accelerator, commit, main_only=True)
    accelerator.wait_for_everyone()
    return next_best
