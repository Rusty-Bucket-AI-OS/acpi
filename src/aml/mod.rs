/*
 * TODO:
 *  - Field reads supporting custom handlers
 *  - Locked fields
 *  - Bank and index fields
 *  - Run `_REG` on supported op region handlers
 *  - Count operations performed and time
 *  - Correct DefStore / DefCopyObject behaviour
 *  - Load and LoadTable
 *  - DefDataRegion
 *  - Notify
 *  - DefMatch
 *
 *  - Method recursion depth?
 *  - Loop timeouts
 *  - Fuzzing and guarantee panic-free interpretation
 */

pub mod namespace;
pub mod object;
pub mod op_region;
pub mod pci_routing;
pub mod resource;

use crate::{
    AcpiError,
    AmlTable,
    Handle,
    Handler,
    PhysicalMapping,
    platform::AcpiPlatform,
    registers::{FixedRegisters, Pm1ControlBit},
    sdt::{SdtHeader, facs::Facs, fadt::Fadt},
};
use alloc::{
    collections::btree_map::BTreeMap,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use bit_field::BitField;
use core::{
    mem,
    slice,
    str::FromStr,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};
use log::{info, trace, warn};
use namespace::{AmlName, Namespace, NamespaceLevelKind};
use object::{
    DeviceStatus,
    FieldFlags,
    FieldUnit,
    FieldUnitKind,
    FieldUpdateRule,
    MethodFlags,
    Object,
    ObjectToken,
    ObjectType,
    ReferenceKind,
    WrappedObject,
};
use op_region::{OpRegion, RegionHandler, RegionSpace};
use pci_types::PciAddress;
use spinning_top::Spinlock;

/// Hardening: limits on what one piece of firmware AML may cost (bounded time and memory). Exceeding any of them is
/// [`AmlError::LimitExceeded`], and the interpreter stays usable for later evaluations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    /// Opcodes (and block ends) executed per top-level `load_table` or `evaluate`.
    pub max_ops: u64,
    /// Method call nesting (AML methods calling methods).
    pub max_depth: usize,
    /// The largest single `Buffer`, `String` or field read the AML may create, in bytes.
    pub max_buffer_bytes: usize,
    /// The most elements of one `Package` or `VarPackage`.
    pub max_package_elements: usize,
    /// The most objects and levels in the namespace.
    pub max_namespace_objects: usize,
    /// Iterations of one `While` loop.
    pub max_loop_iterations: u64,
    /// Nested blocks (scopes, devices, `If`, `While`, packages) inside one method or table.
    pub max_block_nesting: usize,
    /// The longest an AML `Wait` on an event may wait, in milliseconds (including `Wait(.., 0xffff)`, "forever").
    pub max_wait_ms: u64,
    /// The bytes of buffers, strings and field reads one top-level call may create in all.
    pub max_total_bytes: usize,
    /// Operations in flight at once in one method or table (an op left without its operands, in a loop, used to
    /// pile up without bound).
    pub max_in_flight: usize,
    /// The longest one top-level `load_table` or `evaluate` may run, in milliseconds of the host's clock
    /// (`Handler::nanos_since_boot`): a `While` loop of `Sleep`s otherwise holds the interpreter for hours.
    pub max_call_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_ops: 2_000_000,
            max_depth: 32,
            max_buffer_bytes: 1024 * 1024,
            max_package_elements: 4096,
            max_namespace_objects: 100_000,
            max_loop_iterations: 1_000_000,
            max_block_nesting: 256,
            max_wait_ms: 2000,
            max_total_bytes: 16 * 1024 * 1024,
            max_in_flight: 1024,
            max_call_ms: 5000,
        }
    }
}

/// Hardening: which limit an AML stream ran into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Limit {
    Ops,
    MethodDepth,
    Reentrancy,
    BlockNesting,
    /// The call ran past `max_call_ms`.
    Deadline,
    BufferSize,
    PackageElements,
    NamespaceObjects,
    NameDepth,
    LoopIterations,
    TotalBytes,
    InFlight,
}

/// Hardening: what [`Interpreter::initialize_namespace`] did. Failures of one device never stop the others.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct InitReport {
    pub devices_initialized: usize,
    pub sta_failures: usize,
    pub ini_failures: usize,
    pub reg_calls: usize,
    pub reg_failures: usize,
}

/// Hardening: nested evaluations (an evaluation started while another runs, e.g. `_ADR` evaluated for a PCI config
/// field read inside a method) beyond this are refused, so firmware cannot recurse the host's stack.
const MAX_REENTRANT_EVALUATIONS: usize = 4;
/// Hardening: tries to take the firmware's global lock before `Acquire(\_GL)` gives up.
const GLOBAL_LOCK_TRIES: u32 = 1000;

/// `Interpreter` implements a virtual machine for the dynamic AML bytecode. It can be used by a
/// host operating system to load tables containing AML bytecode (generally the DSDT and SSDTs) and
/// will then manage the AML namespace and all objects created during the life of the system.
///
/// Hardening: one thread at a time. The interpreter keeps one stack of method contexts, so a host that may
/// evaluate from several threads wraps it in a lock of its own.
pub struct Interpreter<H>
where
    H: Handler,
{
    handler: H,
    pub namespace: Spinlock<Namespace>,
    pub object_token: Spinlock<ObjectToken>,
    context_stack: Spinlock<Vec<MethodContext>>,
    dsdt_revision: u8,
    // Hardening: `Arc`, so a handler is called with the lock released.
    region_handlers: Spinlock<BTreeMap<RegionSpace, Arc<dyn RegionHandler>>>,

    global_lock_mutex: Handle,
    // Hardening: optional, so the host can build an interpreter without the table side (`new_bare`).
    registers: Option<Arc<FixedRegisters<H>>>,
    facs: Option<PhysicalMapping<H, Facs>>,

    // Hardening: limits and the counters they are checked against.
    limits: Spinlock<Limits>,
    ops_used: AtomicU64,
    bytes_used: AtomicUsize,
    nesting: AtomicUsize,
    /// The host clock's time (ns) the current top-level call must end by.
    deadline: AtomicU64,
}

unsafe impl<H> Send for Interpreter<H> where H: Handler + Send {}
unsafe impl<H> Sync for Interpreter<H> where H: Handler + Send {}

/// The value returned by the `Revision` opcode.
const INTERPRETER_REVISION: u64 = 1;

/// Hardening: a constant path, never failing (the names are valid ASCII). Falls back to the root on the impossible error.
fn static_name(name: &str) -> AmlName {
    AmlName::from_str(name).unwrap_or_else(|_| AmlName::root())
}

impl<H> Interpreter<H>
where
    H: Handler,
{
    /// Construct a new `Interpreter`. This does not load any tables - if you have an `AcpiTables`
    /// already, use [`Interpreter::new_from_tables`] instead.
    pub fn new(
        handler: H,
        dsdt_revision: u8,
        registers: Arc<FixedRegisters<H>>,
        facs: Option<PhysicalMapping<H, Facs>>,
    ) -> Interpreter<H> {
        Self::build(handler, dsdt_revision, Some(registers), facs)
    }

    /// Hardening: an interpreter with no fixed registers and no FACS: the host parses the tables itself and loads the
    /// DSDT and SSDTs with [`Interpreter::load_table`]. `Acquire(\_GL)` then only takes the AML-side mutex
    /// (there is no firmware lock word to take), and releasing it signals nothing to the firmware.
    pub fn new_bare(handler: H, dsdt_revision: u8) -> Interpreter<H> {
        Self::build(handler, dsdt_revision, None, None)
    }

    fn build(
        handler: H,
        dsdt_revision: u8,
        registers: Option<Arc<FixedRegisters<H>>>,
        facs: Option<PhysicalMapping<H, Facs>>,
    ) -> Interpreter<H> {
        info!("Initializing AML interpreter v{}", env!("CARGO_PKG_VERSION"));

        let global_lock_mutex = handler.create_mutex();

        Interpreter {
            handler,
            namespace: Spinlock::new(Namespace::new(global_lock_mutex)),
            object_token: Spinlock::new(unsafe { ObjectToken::create_interpreter_token() }),
            context_stack: Spinlock::new(Vec::new()),
            dsdt_revision,
            region_handlers: Spinlock::new(BTreeMap::new()),
            global_lock_mutex,
            registers,
            facs,
            limits: Spinlock::new(Limits::default()),
            ops_used: AtomicU64::new(0),
            bytes_used: AtomicUsize::new(0),
            nesting: AtomicUsize::new(0),
            deadline: AtomicU64::new(u64::MAX),
        }
    }

    /// Hardening: sets the limits for every later load and evaluation.
    pub fn set_limits(&self, limits: Limits) {
        *self.limits.lock() = limits;
        self.namespace.lock().set_max_objects(limits.max_namespace_objects);
    }

    /// Hardening: the limits in force.
    pub fn limits(&self) -> Limits {
        *self.limits.lock()
    }

    /// Construct a new `Interpreter` with the given set of ACPI tables. This will automatically
    /// load the DSDT and any SSDTs in the supplied [`AcpiTables`].
    pub fn new_from_platform(platform: &AcpiPlatform<H>) -> Result<Interpreter<H>, AcpiError> {
        fn load_table(interpreter: &Interpreter<impl Handler>, table: AmlTable) -> Result<(), AcpiError> {
            // Hardening: a table shorter than its header is refused rather than underflowing the length.
            let Some(stream_len) = (table.length as usize).checked_sub(mem::size_of::<SdtHeader>()) else {
                return Err(AcpiError::Aml(AmlError::MalformedStream));
            };
            let mapping = unsafe {
                interpreter.handler.map_physical_region::<SdtHeader>(table.phys_address, table.length as usize)
            };
            let stream = unsafe {
                slice::from_raw_parts(
                    mapping.virtual_start.as_ptr().byte_add(mem::size_of::<SdtHeader>()) as *const u8,
                    stream_len,
                )
            };
            interpreter.load_table(stream).map_err(AcpiError::Aml)?;
            Ok(())
        }

        let registers = platform.registers.clone();
        let facs = {
            platform.tables.find_table::<Fadt>().and_then(|fadt| fadt.facs_address().ok()).map(
                |facs_address| unsafe {
                    platform.handler.map_physical_region(facs_address, mem::size_of::<Facs>())
                },
            )
        };

        let dsdt = platform.tables.dsdt()?;
        let interpreter = Interpreter::new(platform.handler.clone(), dsdt.revision, registers, facs);
        load_table(&interpreter, dsdt)?;

        for ssdt in platform.tables.ssdts() {
            load_table(&interpreter, ssdt)?;
        }

        Ok(interpreter)
    }

    /// Load the supplied byte stream as an AML table. This should be only the encoded AML stream -
    /// not the header at the start of a table. If you've used [`Interpreter::new_from_tables`],
    /// you'll likely not need to load any tables manually.
    pub fn load_table(&self, stream: &[u8]) -> Result<(), AmlError> {
        let context = unsafe { MethodContext::new_from_table(stream) };
        self.do_execute_method(context)?;
        Ok(())
    }

    /// Evaluate an object at the given path in the namespace. If the object is a method, this
    /// invokes the method with the given set of arguments.
    pub fn evaluate(&self, path: AmlName, args: Vec<WrappedObject>) -> Result<WrappedObject, AmlError> {
        trace!("Invoking AML method: {}", path);

        let object = self.namespace.lock().get(path.clone())?.clone();
        match &*object {
            Object::Method { .. } => {
                self.namespace.lock().add_level(path.clone(), NamespaceLevelKind::MethodLocals)?;
                let context = match MethodContext::new_from_method(object.clone(), args, path.clone()) {
                    Ok(context) => context,
                    Err(err) => {
                        // Hardening: the level added above is not left behind.
                        let _ = self.namespace.lock().remove_level(path);
                        return Err(err);
                    }
                };
                self.do_execute_method(context)
            }
            Object::NativeMethod { f, .. } => f(&args),
            _ => Ok(object),
        }
    }

    pub fn evaluate_if_present(
        &self,
        path: AmlName,
        args: Vec<WrappedObject>,
    ) -> Result<Option<WrappedObject>, AmlError> {
        match self.evaluate(path.clone(), args) {
            Ok(result) => Ok(Some(result)),
            Err(AmlError::ObjectDoesNotExist(not_present)) => {
                if path == not_present {
                    Ok(None)
                } else {
                    Err(AmlError::ObjectDoesNotExist(not_present))
                }
            }
            // Hardening: a path whose parent level does not exist is just as absent.
            Err(AmlError::LevelDoesNotExist(_)) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// Hardening: installs the handler for one region space. A second handler for the same space is an error (it used to
    /// be an `assert!`). Field reads and writes of that space go through it, the native spaces too if one is
    /// installed for them.
    pub fn install_region_handler<RH>(&self, space: RegionSpace, handler: RH) -> Result<(), AmlError>
    where
        RH: RegionHandler + 'static,
    {
        let mut handlers = self.region_handlers.lock();
        if handlers.contains_key(&space) {
            return Err(AmlError::RegionHandlerAlreadyInstalled(space));
        }
        handlers.insert(space, Arc::new(handler));
        Ok(())
    }

    fn region_handler(&self, space: RegionSpace) -> Option<Arc<dyn RegionHandler>> {
        self.region_handlers.lock().get(&space).cloned()
    }

    /// Initialize the namespace - this should be called after all tables have been loaded and
    /// operation region handlers registered. Specifically, it will call relevant `_STA`, `_INI`,
    /// and `_REG` methods.
    ///
    /// Hardening: one device's failing `_STA` or `_INI` never stops the others; the report counts them. `_REG(space, 1)` is
    /// run for every device that has an operation region of a space with an installed handler.
    pub fn initialize_namespace(&self) -> InitReport {
        let mut report = InitReport::default();
        /*
         * This should match the initialization order of ACPICA and uACPI.
         */
        if let Err(err) = self.evaluate_if_present(static_name("\\_INI"), vec![]) {
            warn!("Invoking \\_INI failed: {:?}", err);
            report.ini_failures += 1;
        }
        if let Err(err) = self.evaluate_if_present(static_name("\\_SB._INI"), vec![]) {
            warn!("Invoking \\_SB._INI failed: {:?}", err);
            report.ini_failures += 1;
        }

        self.run_reg_methods(&mut report);

        /*
         * We can now initialize each device in the namespace. For each device, we evaluate `_STA`,
         * which indicates if the device is present and functional. If this method does not exist,
         * we assume the device should be initialized.
         *
         * We then evaluate `_INI` for the device. This can dynamically populate objects such as
         * `_ADR`, `_CID`, `_HID`, `_SUN`, and `_UID`, and so is necessary before further
         * operation.
         */
        let mut namespace = self.namespace.lock().clone();
        let init_status = namespace.traverse(|path, level| {
            match level.kind {
                NamespaceLevelKind::Device
                | NamespaceLevelKind::Processor
                | NamespaceLevelKind::ThermalZone
                | NamespaceLevelKind::PowerResource => {
                    let Ok(sta) = static_name("_STA").resolve(path) else {
                        report.sta_failures += 1;
                        return Ok(false);
                    };
                    let should_initialize = match self.evaluate_if_present(sta, vec![]) {
                        Ok(Some(result)) => match *result {
                            Object::Integer(result) => {
                                let status = DeviceStatus(result);
                                status.present() && status.functioning()
                            }
                            // Hardening: was `panic!()`.
                            _ => {
                                report.sta_failures += 1;
                                false
                            }
                        },
                        Ok(None) => true,
                        Err(err) => {
                            warn!("Failed to evaluate _STA for device {}: {:?}", path, err);
                            report.sta_failures += 1;
                            false
                        }
                    };

                    if should_initialize {
                        report.devices_initialized += 1;
                        let ini = static_name("_INI").resolve(path);
                        if let Err(err) = ini.and_then(|ini| self.evaluate_if_present(ini, vec![])) {
                            warn!("Failed to evaluate _INI for device {}: {:?}", path, err);
                            report.ini_failures += 1;
                        }
                        Ok(true)
                    } else {
                        /*
                         * If this device should not be initialized, don't initialize it's children.
                         */
                        Ok(false)
                    }
                }
                _ => Ok(true),
            }
        });
        if let Err(err) = init_status {
            warn!("Error while traversing namespace for devices: {:?}", err);
        }

        info!("Initialized {} devices", report.devices_initialized);
        report
    }

    /// Hardening: `_REG(space, 1)` for each device that owns an operation region of a space with a handler.
    fn run_reg_methods(&self, report: &mut InitReport) {
        let spaces: Vec<RegionSpace> = self.region_handlers.lock().keys().copied().collect();
        if spaces.is_empty() {
            return;
        }
        let mut targets: Vec<(AmlName, RegionSpace)> = Vec::new();
        let namespace = self.namespace.lock().clone();
        namespace.for_each_object(|object| {
            if let Object::OpRegion(ref region) = **object
                && spaces.contains(&region.space)
                && !targets.iter().any(|(p, s)| *p == region.parent_device_path && *s == region.space)
            {
                targets.push((region.parent_device_path.clone(), region.space));
            }
        });
        for (device, space) in targets {
            let Ok(reg) = static_name("_REG").resolve(&device) else {
                continue;
            };
            let args = vec![Object::Integer(u8::from(space) as u64).wrap(), Object::Integer(1).wrap()];
            match self.evaluate_if_present(reg, args) {
                Ok(Some(_)) => report.reg_calls += 1,
                Ok(None) => {}
                Err(err) => {
                    warn!("_REG for {} failed: {:?}", device, err);
                    report.reg_failures += 1;
                }
            }
        }
    }

    pub fn acquire_global_lock(&self, timeout: u16) -> Result<(), AmlError> {
        self.handler.acquire(self.global_lock_mutex, timeout)?;

        // Now we've acquired the AML-side mutex, acquire the hardware side
        // Hardening: bounded, and the AML-side mutex is kept while trying (it was released on every try, and never taken
        // again).
        for _ in 0..GLOBAL_LOCK_TRIES {
            if self.try_do_acquire_firmware_lock() {
                return Ok(());
            }
            /*
             * The lock is owned by the firmware. We have set the pending bit - we now need to
             * wait for the firmware to signal it has released the lock.
             */
            self.handler.stall(10);
        }
        self.handler.release(self.global_lock_mutex);
        Err(AmlError::MutexAcquireTimeout)
    }

    /// Attempt to acquire the firmware lock, setting the owned bit if the lock is free. If the
    /// lock is not free, sets the pending bit to instruct the firmware to alert us when we can
    /// attempt to take ownership of the lock again. Returns `true` if we now have ownership of the
    /// lock, and `false` if we need to wait for firmware to release it.
    fn try_do_acquire_firmware_lock(&self) -> bool {
        let Some(facs) = &self.facs else { return true };
        loop {
            let global_lock = facs.global_lock.load(Ordering::Relaxed);
            let is_owned = global_lock.get_bit(1);

            /*
             * Compute the new value: either the lock is already owned, and we need to set the
             * pending bit and wait, or we can acquire ownership of the lock now. Either way, we
             * unconditionally set the owned bit and set the pending bit if the lock is already
             * owned.
             */
            let mut new_value = global_lock;
            new_value.set_bit(0, is_owned);
            new_value.set_bit(1, true);

            if facs
                .global_lock
                .compare_exchange(global_lock, new_value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break !is_owned;
            }
        }
    }

    pub fn release_global_lock(&self) -> Result<(), AmlError> {
        let is_pending = self.do_release_firmware_lock();
        self.handler.release(self.global_lock_mutex);
        if is_pending && let Some(registers) = &self.registers {
            // Hardening: was `.unwrap()`.
            registers
                .pm1_control_registers
                .set_bit(Pm1ControlBit::GlobalLockRelease, true)
                .map_err(|_| AmlError::RegisterAccessFailed)?;
        }
        Ok(())
    }

    /// Atomically release the owned and pending bits of the global lock. Returns whether the
    /// pending bit was set (this means the firmware is waiting to acquire the lock, and should be
    /// informed we're finished with it).
    fn do_release_firmware_lock(&self) -> bool {
        let Some(facs) = &self.facs else { return false };
        loop {
            let global_lock = facs.global_lock.load(Ordering::Relaxed);
            let is_pending = global_lock.get_bit(0);
            let mut new_value = global_lock;
            new_value.set_bit(0, false);
            new_value.set_bit(1, false);

            if facs
                .global_lock
                .compare_exchange(global_lock, new_value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break is_pending;
            }
        }
    }

    /// Hardening: runs a table or a method to its end. Bounds how deeply evaluations nest and how many ops one top-level
    /// call runs; on any error, every context this call pushed is taken off the shared stack again and their
    /// method levels are removed from the namespace, so the interpreter stays usable (before, a failed call left
    /// stale contexts behind, the next `Return` popped one, and a table's context pointed at a stream that was gone).
    fn do_execute_method(&self, mut context: MethodContext) -> Result<WrappedObject, AmlError> {
        let depth = self.nesting.fetch_add(1, Ordering::AcqRel);
        if depth == 0 {
            self.ops_used.store(0, Ordering::Release);
            self.bytes_used.store(0, Ordering::Release);
            let budget = self.limits().max_call_ms.saturating_mul(1_000_000);
            self.deadline.store(self.handler.nanos_since_boot().saturating_add(budget), Ordering::Release);
        }
        let result = if depth >= MAX_REENTRANT_EVALUATIONS {
            Err(AmlError::LimitExceeded(Limit::Reentrancy))
        } else {
            let base = self.context_stack.lock().len();
            let result = self.run_context(&mut context, base);
            if result.is_err() {
                self.unwind(context, base);
            }
            result
        };
        self.nesting.fetch_sub(1, Ordering::AcqRel);
        result
    }

    /// Hardening: after an error: drops the failed context and every context above `base`, removing their method levels.
    fn unwind(&self, context: MethodContext, base: usize) {
        let mut dropped = vec![context];
        {
            let mut stack = self.context_stack.lock();
            while stack.len() > base {
                if let Some(old) = stack.pop() {
                    dropped.push(old);
                }
            }
        }
        for old in dropped {
            if let Some(scope) = old.method_scope() {
                let _ = self.namespace.lock().remove_level(scope);
            }
        }
    }

    /// Hardening: a size the AML wants to create, before anything is allocated: against the limit for one object and the
    /// budget of the whole top-level call.
    fn check_size(&self, bytes: usize) -> Result<(), AmlError> {
        let limits = self.limits();
        if bytes > limits.max_buffer_bytes {
            return Err(AmlError::LimitExceeded(Limit::BufferSize));
        }
        let used = self.bytes_used.fetch_add(bytes, Ordering::AcqRel).saturating_add(bytes);
        if used > limits.max_total_bytes { Err(AmlError::LimitExceeded(Limit::TotalBytes)) } else { Ok(()) }
    }

    /// Hardening: one more op of the current top-level call.
    fn count_op(&self, limits: &Limits) -> Result<(), AmlError> {
        let used = self.ops_used.fetch_add(1, Ordering::AcqRel) + 1;
        if used > limits.max_ops {
            return Err(AmlError::LimitExceeded(Limit::Ops));
        }
        self.check_deadline()
    }

    /// Hardening: whether the current top-level call is still within its time (`Limits::max_call_ms`).
    fn check_deadline(&self) -> Result<(), AmlError> {
        if self.handler.nanos_since_boot() > self.deadline.load(Ordering::Acquire) {
            Err(AmlError::LimitExceeded(Limit::Deadline))
        } else {
            Ok(())
        }
    }

    /// Hardening: milliseconds left before the current call's deadline (a `Sleep` never sleeps past it).
    fn ms_left(&self) -> u64 {
        self.deadline.load(Ordering::Acquire).saturating_sub(self.handler.nanos_since_boot()) / 1_000_000
    }

    /// Hardening: the caller's context, if this top-level call pushed it (contexts below `base` belong to an outer call).
    fn pop_context(&self, base: usize) -> Option<MethodContext> {
        let mut stack = self.context_stack.lock();
        if stack.len() > base { stack.pop() } else { None }
    }

    fn run_context(&self, context: &mut MethodContext, base: usize) -> Result<WrappedObject, AmlError> {
        let limits = self.limits();
        /*
         * This is the main loop that executes operations. Every op is handled at the top-level of
         * the loop to prevent pathological stack growth from nested operations.
         *
         * The loop has three main stages:
         *   1) Check if any in-flight operations are ready to be executed (i.e. have collected all
         *      their arguments). An operation completing may contribute the last required argument
         *      of the one above, so this is repeated for as many operations as are ready to be
         *      retired.
         *   2) Look at the next opcode in the stream. If we've run out of opcodes in the current
         *      block, run logic to determine where in the stream we should move to next. Special
         *      logic at this level handles things like moving in/out of package definitions, and
         *      performing control flow.
         *   3) When the next opcode is determined, use it to interpret the next portion of the
         *      stream. If that is data, the correct number of bytes can be consumed and
         *      contributed to the current in-flight operation. If it's an opcode, a new in-flight
         *      operation is started, and we go round the loop again.
         *
         * This scheme is what allows the interpreter to use a loop that somewhat resembles a
         * traditional fast bytecode VM, but also provides enough flexibility to handle the
         * quirkier parts of the AML grammar, particularly the left-to-right encoding of operands.
         */
        loop {
            // Hardening: every iteration counts against the top-level call's op budget, and the ops in flight are bounded.
            self.count_op(&limits)?;
            if context.in_flight.len() > limits.max_in_flight {
                return Err(AmlError::LimitExceeded(Limit::InFlight));
            }
            /*
             * First, see if we've gathered enough arguments to complete some in-flight operations.
             */
            while let Some(op) = context.in_flight.pop_if(|op| op.arguments.len() == op.expected_arguments) {
                match op.op {
                    Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
                    | Opcode::ShiftLeft
                    | Opcode::ShiftRight
                    | Opcode::Mod
                    | Opcode::Nand
                    | Opcode::And
                    | Opcode::Or
                    | Opcode::Nor
                    | Opcode::Xor => {
                        self.do_binary_maths(context, op)?;
                    }
                    Opcode::Not | Opcode::FindSetLeftBit | Opcode::FindSetRightBit => {
                        self.do_unary_maths(context, op)?;
                    }
                    Opcode::Increment | Opcode::Decrement => {
                        let [Argument::Object(operand)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let token = self.object_token.lock();
                        // SAFETY: the object token is held, so no other context mutates objects, and this op holds
                        // no other reference into `operand` while the integer is changed in place.
                        let Object::Integer(operand) = (unsafe { operand.gain_mut(&token) }) else {
                            Err(AmlError::ObjectNotOfExpectedType {
                                expected: ObjectType::Integer,
                                got: operand.typ(),
                            })?
                        };

                        let new_value = if op.op == Opcode::Increment {
                            operand.wrapping_add(1)
                        } else {
                            operand.wrapping_sub(1)
                        };

                        *operand = new_value;
                        context.retire_op(op);
                    }
                    Opcode::LAnd
                    | Opcode::LOr
                    | Opcode::LNot
                    | Opcode::LNotEqual
                    | Opcode::LLessEqual
                    | Opcode::LGreaterEqual
                    | Opcode::LEqual
                    | Opcode::LGreater
                    | Opcode::LLess => {
                        self.do_logical_op(context, op)?;
                    }
                    Opcode::ToBuffer => self.do_to_buffer(context, op)?,
                    Opcode::ToInteger => self.do_to_integer(context, op)?,
                    Opcode::ToString => self.do_to_string(context, op)?,
                    Opcode::ToDecimalString | Opcode::ToHexString => self.do_to_dec_hex_string(context, op)?,
                    Opcode::Mid => self.do_mid(context, op)?,
                    Opcode::Concat => self.do_concat(context, op)?,
                    Opcode::ConcatRes => {
                        let [Argument::Object(source1), Argument::Object(source2), target] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let source1 = source1.as_buffer()?;
                        let source2 = source2.as_buffer()?;
                        self.check_size(source1.len().saturating_add(source2.len()).saturating_add(2))?;
                        let result = {
                            let mut buffer = Vec::from(source1);
                            buffer.extend_from_slice(source2);
                            // Add a new end-tag
                            buffer.push(0x78);
                            // Don't calculate the new real checksum - just use 0
                            buffer.push(0x00);
                            Object::Buffer(buffer).wrap()
                        };
                        // TODO: use potentially-updated result for return value here
                        self.do_store(target, result.clone())?;
                        context.contribute_arg(Argument::Object(result));
                        context.retire_op(op);
                    }
                    Opcode::Reset => {
                        let [Argument::Object(sync_object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let sync_object = sync_object.clone().unwrap_reference();

                        if let Object::Event(ref counter) = *sync_object {
                            counter.store(0, Ordering::Release);
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::ResetEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::Signal => {
                        let [Argument::Object(sync_object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let sync_object = sync_object.clone().unwrap_reference();

                        if let Object::Event(ref counter) = *sync_object {
                            counter.fetch_add(1, Ordering::AcqRel);
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::SignalEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::Wait => {
                        let [Argument::Object(sync_object), Argument::Object(timeout)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let sync_object = sync_object.clone().unwrap_reference();
                        // Hardening: "forever" (0xffff) and every other timeout are bounded by `max_wait_ms`.
                        let timeout = u64::min(timeout.as_integer()?, limits.max_wait_ms);

                        if let Object::Event(ref counter) = *sync_object {
                            /*
                             * `Wait` returns a non-zero value if a timeout occurs and the event
                             * was not signaled, and zero if it was. Timeout is specified in
                             * milliseconds, should relinquish processor control (we use
                             * `Handler::sleep` to do so) and a value of `0xffff` specifies that
                             * the operation should wait indefinitely.
                             */
                            let mut remaining_sleep = timeout;
                            let mut timed_out = true;

                            'signaled: while remaining_sleep > 0 {
                                loop {
                                    /*
                                     * Try to decrement the counter. If it's zero after a load, we
                                     * haven't been signalled and should wait for a bit. If it's
                                     * non-zero, we were signalled and should stop waiting.
                                     */
                                    let value = counter.load(Ordering::Acquire);
                                    if value == 0 {
                                        break;
                                    }
                                    if counter
                                        .compare_exchange(value, value - 1, Ordering::AcqRel, Ordering::Acquire)
                                        .is_ok()
                                    {
                                        timed_out = false;
                                        break 'signaled;
                                    }
                                }

                                let to_sleep = u64::min(remaining_sleep, 10);
                                remaining_sleep -= to_sleep;
                                self.handler.sleep(to_sleep);
                            }

                            context.contribute_arg(Argument::Object(
                                Object::Integer(if timed_out { u64::MAX } else { 0 }).wrap(),
                            ));
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::WaitEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::FromBCD => self.do_from_bcd(context, op)?,
                    Opcode::ToBCD => self.do_to_bcd(context, op)?,
                    Opcode::Name => {
                        let [Argument::Namestring(name), Argument::Object(object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };

                        let name = name.resolve(&context.current_scope)?;
                        self.namespace.lock().insert(name, object.clone())?;
                        context.retire_op(op);
                    }
                    Opcode::Fatal => {
                        let [Argument::ByteData(typ), Argument::DWordData(code), Argument::Object(arg)] =
                            &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let arg = arg.as_integer()?;
                        self.handler.handle_fatal_error(*typ, *code, arg);
                        // Hardening: the firmware asked to stop: this evaluation ends with an error (the host decides).
                        return Err(AmlError::FatalOpcode { typ: *typ, code: *code, arg });
                    }
                    Opcode::OpRegion => {
                        let [
                            Argument::Namestring(name),
                            Argument::ByteData(region_space),
                            Argument::Object(region_offset),
                            Argument::Object(region_length),
                        ] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };

                        let region_offset = region_offset.clone().unwrap_transparent_reference();
                        let region_length = region_length.clone().unwrap_transparent_reference();

                        let region = Object::OpRegion(OpRegion {
                            space: RegionSpace::from(*region_space),
                            base: region_offset.as_integer()?,
                            length: region_length.as_integer()?,
                            parent_device_path: context.current_scope.clone(),
                        });
                        self.namespace.lock().insert(name.resolve(&context.current_scope)?, region.wrap())?;
                        context.retire_op(op);
                    }
                    Opcode::DataRegion => {
                        let [
                            Argument::Namestring(name),
                            Argument::Object(signature),
                            Argument::Object(oem_id),
                            Argument::Object(oem_table_id),
                        ] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let _signature = signature.as_string()?;
                        let _oem_id = oem_id.as_string()?;
                        let _oem_table_id = oem_table_id.as_string()?;

                        // TODO: once this is integrated into the rest of the crate, load the table
                        log::warn!(
                            "DefDataRegion encountered in AML! We don't actually support these - produced region will be incorrect"
                        );

                        let region = Object::OpRegion(OpRegion {
                            space: RegionSpace::SystemMemory,
                            base: 0,
                            length: 0,
                            parent_device_path: context.current_scope.clone(),
                        });
                        self.namespace.lock().insert(name.resolve(&context.current_scope)?, region.wrap())?;
                        context.retire_op(op);
                    }
                    Opcode::Buffer => {
                        let [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(pkg_length),
                            Argument::Object(buffer_size),
                        ] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let buffer_size = buffer_size.clone().unwrap_transparent_reference().as_integer()?;

                        // Hardening: the size is checked before anything is allocated, the initializer is bounded by the
                        // stream, and a longer initializer than the size makes the buffer that long (ACPI 19.6.10).
                        let buffer_len = context.remaining(*start_pc, *pkg_length)?;
                        let initializer = context.take(buffer_len)?;
                        let buffer_size = usize::try_from(buffer_size).unwrap_or(usize::MAX).max(buffer_len);
                        self.check_size(buffer_size)?;
                        let mut buffer = vec![0; buffer_size];
                        buffer[0..buffer_len].copy_from_slice(initializer);

                        context.contribute_arg(Argument::Object(Object::Buffer(buffer).wrap()));
                        context.retire_op(op);
                    }
                    Opcode::Package => {
                        let mut elements = Vec::with_capacity(op.arguments.len());
                        for arg in &op.arguments {
                            let Argument::Object(object) = arg else {
                                return Err(AmlError::MalformedStream);
                            };
                            elements.push(object.clone());
                        }

                        /*
                         * We can end up completing a package's in-flight op in two circumstances:
                         *    - If the correct number of elements are supplied, we end up here
                         *      first, and then later in the block's finishing logic.
                         *    - If less elements are supplied, we end up in the block's finishing
                         *      logic to add some `Uninitialized`s, then go round again to complete
                         *      the in-flight operation.
                         *
                         * To make these consistent, we always remove the block here, making sure
                         * we've finished it as a sanity check.
                         */
                        // Hardening: these were `assert!`s and an `unwrap`. A package whose block is not the current one is
                        // malformed; bytes past its declared elements are skipped.
                        if context.current_block.kind != BlockKind::Package {
                            return Err(AmlError::MalformedStream);
                        }
                        context.pop_block()?;
                        context.contribute_arg(Argument::Object(Object::Package(elements).wrap()));
                        context.retire_op(op);
                    }
                    Opcode::VarPackage => {
                        let Some(Argument::Object(total_elements)) = op.arguments.first() else {
                            return Err(AmlError::MalformedStream);
                        };
                        let total_elements = total_elements.clone().unwrap_transparent_reference().as_integer()?;
                        let total_elements = check_elements(&limits, total_elements)?;

                        let mut elements = Vec::with_capacity(total_elements);
                        for arg in op.arguments.iter().skip(1).take(total_elements) {
                            let Argument::Object(object) = arg else {
                                return Err(AmlError::MalformedStream);
                            };
                            elements.push(object.clone());
                        }

                        /*
                         * As above, we always remove the block here after the in-flight op has
                         * been retired.
                         */
                        if context.current_block.kind != BlockKind::VarPackage {
                            return Err(AmlError::MalformedStream);
                        }
                        context.pop_block()?;
                        context.contribute_arg(Argument::Object(Object::Package(elements).wrap()));
                        context.retire_op(op);
                    }
                    Opcode::If => {
                        let [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(then_length),
                            Argument::Object(predicate),
                        ] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let predicate = predicate.as_integer()?;
                        let remaining_then_length = context.remaining(*start_pc, *then_length)?;

                        if predicate > 0 {
                            context.start_new_block(BlockKind::IfThenBranch, remaining_then_length, &limits)?;
                        } else {
                            context.skip(remaining_then_length)?;

                            /*
                             * Skip over the prolog to the else branch if present. Also handle if
                             * there are no more bytes to peek - the `If` op could be the last op
                             * in a block.
                             */
                            const DEF_ELSE_OP: u8 = 0xa1;
                            match context.peek() {
                                Ok(DEF_ELSE_OP) => {
                                    context.next()?;
                                    let _else_length = context.pkglength()?;
                                }
                                Ok(_) => (),
                                Err(AmlError::RunOutOfStream) => (),
                                Err(other) => Err(other)?,
                            }
                        }
                        context.retire_op(op);
                    }
                    opcode @ Opcode::CreateBitField
                    | opcode @ Opcode::CreateByteField
                    | opcode @ Opcode::CreateWordField
                    | opcode @ Opcode::CreateDWordField
                    | opcode @ Opcode::CreateQWordField => {
                        let [Argument::Object(buffer), Argument::Object(index)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let name = context.namestring()?;
                        let index = index.as_integer()?;
                        // Hardening: `index * 8` could overflow; a field past the buffer is refused when it is created, and again when used.
                        let length = match opcode {
                            Opcode::CreateBitField => 1,
                            Opcode::CreateByteField => 8,
                            Opcode::CreateWordField => 16,
                            Opcode::CreateDWordField => 32,
                            _ => 64,
                        };
                        let offset =
                            if opcode == Opcode::CreateBitField { Some(index) } else { index.checked_mul(8) };
                        let offset = offset
                            .and_then(|offset| usize::try_from(offset).ok())
                            .ok_or(AmlError::IndexOutOfBounds)?;
                        field_in_buffer(buffer, offset, length)?;
                        self.namespace.lock().insert(
                            name.resolve(&context.current_scope)?,
                            Object::BufferField { buffer: buffer.clone(), offset, length }.wrap(),
                        )?;
                        context.retire_op(op);
                    }
                    Opcode::CreateField => {
                        let [Argument::Object(buffer), Argument::Object(bit_index), Argument::Object(num_bits)] =
                            &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let name = context.namestring()?;
                        let bit_index =
                            usize::try_from(bit_index.as_integer()?).map_err(|_| AmlError::IndexOutOfBounds)?;
                        let num_bits =
                            usize::try_from(num_bits.as_integer()?).map_err(|_| AmlError::IndexOutOfBounds)?;
                        // Hardening: a field past its buffer (or of no bits) is refused when it is created.
                        field_in_buffer(buffer, bit_index, num_bits)?;

                        self.namespace.lock().insert(
                            name.resolve(&context.current_scope)?,
                            Object::BufferField { buffer: buffer.clone(), offset: bit_index, length: num_bits }
                                .wrap(),
                        )?;
                        context.retire_op(op);
                    }
                    Opcode::Store => {
                        let [Argument::Object(object), target] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        self.do_store(target, object.clone())?;
                        context.retire_op(op);
                    }
                    Opcode::RefOf => {
                        let [Argument::Object(object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let reference =
                            Object::Reference { kind: ReferenceKind::RefOf, inner: object.clone() }.wrap();
                        context.contribute_arg(Argument::Object(reference));
                        context.retire_op(op);
                    }
                    Opcode::CondRefOf => {
                        let [Argument::Object(object), target] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let result = if let Object::Reference { kind: ReferenceKind::Unresolved, .. } = **object {
                            Object::Integer(0)
                        } else {
                            let reference =
                                Object::Reference { kind: ReferenceKind::RefOf, inner: object.clone() }.wrap();
                            self.do_store(target, reference)?;
                            Object::Integer(u64::MAX)
                        };
                        context.contribute_arg(Argument::Object(result.wrap()));
                        context.retire_op(op);
                    }
                    Opcode::DerefOf => {
                        let [Argument::Object(object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        // Hardening: `typ()` reports what a reference refers to, so a reference never matched here.
                        let result = if let Object::Reference { .. } = **object {
                            object.clone().unwrap_reference()
                        } else if object.typ() == ObjectType::String {
                            let path = AmlName::from_str(&object.as_string()?)?.resolve(&context.current_scope)?;
                            self.namespace.lock().get(path)?.clone()
                        } else {
                            return Err(AmlError::ObjectNotOfExpectedType {
                                expected: ObjectType::Reference,
                                got: object.typ(),
                            });
                        };
                        context.contribute_arg(Argument::Object(result));
                        context.retire_op(op);
                    }
                    Opcode::Sleep => {
                        let [Argument::Object(msec)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        // Hardening: never past the call's deadline; the next op then ends the call.
                        self.handler.sleep(u64::min(msec.as_integer()?, self.ms_left().saturating_add(1)));
                        self.check_deadline()?;
                        context.retire_op(op);
                    }
                    Opcode::Stall => {
                        let [Argument::Object(usec)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        self.handler.stall(usec.as_integer()?);
                        context.retire_op(op);
                    }
                    Opcode::Acquire => {
                        let [Argument::Object(mutex)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let Object::Mutex { mutex, sync_level: _ } = **mutex else {
                            Err(AmlError::InvalidOperationOnObject { op: Operation::Acquire, typ: mutex.typ() })?
                        };
                        let timeout = context.next_u16()?;

                        // TODO: should we do something with the sync level??
                        if mutex == self.global_lock_mutex {
                            self.acquire_global_lock(timeout)?;
                        } else {
                            self.handler.acquire(mutex, timeout)?;
                        }

                        context.retire_op(op);
                    }
                    Opcode::Release => {
                        let [Argument::Object(mutex)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let Object::Mutex { mutex, sync_level: _ } = **mutex else {
                            Err(AmlError::InvalidOperationOnObject { op: Operation::Release, typ: mutex.typ() })?
                        };

                        // TODO: should we do something with the sync level??
                        if mutex == self.global_lock_mutex {
                            self.release_global_lock()?;
                        } else {
                            self.handler.release(mutex);
                        }

                        context.retire_op(op);
                    }
                    Opcode::InternalMethodCall => {
                        let [Argument::Object(method), Argument::Namestring(method_scope), ..] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };

                        let mut args = Vec::with_capacity(op.arguments.len().saturating_sub(2));
                        for arg in &op.arguments[2..] {
                            let Argument::Object(arg) = arg else {
                                return Err(AmlError::MalformedStream);
                            };
                            args.push(arg.clone());
                        }

                        if let Object::Method { .. } = **method {
                            // Hardening: bounded method nesting.
                            if self.context_stack.lock().len() >= limits.max_depth {
                                return Err(AmlError::LimitExceeded(Limit::MethodDepth));
                            }
                            let new_context =
                                MethodContext::new_from_method(method.clone(), args, method_scope.clone())?;
                            self.namespace
                                .lock()
                                .add_level(method_scope.clone(), NamespaceLevelKind::MethodLocals)?;
                            let old_context = mem::replace(context, new_context);
                            self.context_stack.lock().push(old_context);
                            context.retire_op(op);
                        } else if let Object::NativeMethod { ref f, .. } = **method {
                            let result = f(&args)?;
                            context.contribute_arg(Argument::Object(result));
                        } else {
                            return Err(AmlError::ObjectNotOfExpectedType {
                                expected: ObjectType::Method,
                                got: method.typ(),
                            });
                        }
                    }
                    Opcode::Return => {
                        let [Argument::Object(object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let object = object.clone().unwrap_transparent_reference();

                        // Hardening: the returning method's level is removed (it used to stay behind), and only contexts
                        // this call pushed are popped.
                        if let Some(scope) = context.method_scope() {
                            let _ = self.namespace.lock().remove_level(scope);
                        }
                        if let Some(last) = self.pop_context(base) {
                            *context = last;
                            context.contribute_arg(Argument::Object(object.clone()));
                            context.retire_op(op);
                        } else {
                            /*
                             * If this is the top-most context, this is a `Return` from the actual
                             * method.
                             */
                            return Ok(object.clone());
                        }
                    }
                    Opcode::ObjectType => {
                        let [Argument::Object(object)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        // TODO: this should technically support scopes as well - this is less easy
                        // (they should return `0`)
                        let typ = match object.typ() {
                            ObjectType::Uninitialized => 0,
                            ObjectType::Integer => 1,
                            ObjectType::String => 2,
                            ObjectType::Buffer => 3,
                            ObjectType::Package => 4,
                            ObjectType::FieldUnit => 5,
                            ObjectType::Device => 6,
                            ObjectType::Event => 7,
                            ObjectType::Method => 8,
                            ObjectType::Mutex => 9,
                            ObjectType::OpRegion => 10,
                            ObjectType::PowerResource => 11,
                            ObjectType::Processor => 12,
                            ObjectType::ThermalZone => 13,
                            ObjectType::BufferField => 14,
                            // XXX: 15 is reserved
                            ObjectType::Debug => 16,
                            // Hardening: were `panic!()` and `todo!()`.
                            ObjectType::Reference | ObjectType::RawDataBuffer => 0,
                        };

                        context.contribute_arg(Argument::Object(Object::Integer(typ).wrap()));
                        context.retire_op(op);
                    }
                    Opcode::SizeOf => self.do_size_of(context, op)?,
                    Opcode::Index => self.do_index(context, op)?,
                    Opcode::BankField => {
                        let [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(pkg_length),
                            Argument::Namestring(region_name),
                            Argument::Namestring(bank_name),
                            Argument::Object(bank_value),
                        ] = &op.arguments[..]
                        else {
                            return Err(AmlError::MalformedStream);
                        };
                        let bank_value = bank_value.as_integer()?;
                        let field_flags = context.next()?;

                        let (region, bank) = {
                            let namespace = self.namespace.lock();
                            let (_, region) = namespace.search(region_name, &context.current_scope)?;
                            let (_, bank) = namespace.search(bank_name, &context.current_scope)?;
                            (region, bank)
                        };

                        let kind = FieldUnitKind::Bank { region, bank, bank_value };
                        self.parse_field_list(context, kind, *start_pc, *pkg_length, field_flags)?;
                        context.retire_op(op);
                    }
                    Opcode::While => {
                        /*
                         * We've just evaluated the predicate for an iteration of a while loop. If
                         * false, skip over the rest of the loop, otherwise carry on.
                         */
                        let [Argument::Object(predicate)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let predicate = predicate.as_integer()?;

                        if predicate == 0 {
                            // Exit from the while loop by skipping out of the current block
                            context.pop_block()?;
                            context.retire_op(op);
                        }
                    }
                    Opcode::Notify => {
                        // Hardening: was the catch-all `panic!` below. The host hears of it with the object's resolved path.
                        let [Argument::Object(_object), Argument::Object(value)] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        let value = value.clone().unwrap_transparent_reference().as_integer()?;
                        if let Some(path) = &op.target_name {
                            self.handler.handle_notify(path, value);
                        } else {
                            warn!(
                                "Notify({:#x}) of an object with no name (a local or an argument): not delivered",
                                value
                            );
                        }
                        context.retire_op(op);
                    }
                    Opcode::CopyObject => {
                        // Hardening: was `todo!()`. Done as a store: the target takes the source's value.
                        let [Argument::Object(object), target] = &op.arguments[..] else {
                            return Err(AmlError::MalformedStream);
                        };
                        self.do_store(target, object.clone())?;
                        context.contribute_arg(Argument::Object(object.clone()));
                        context.retire_op(op);
                    }
                    other => {
                        // Hardening: was `panic!("Unexpected operation has created in-flight op!")`.
                        warn!("AML: an in-flight {:?} has no completion", other);
                        return Err(AmlError::MalformedStream);
                    }
                }
            }

            /*
             * Now that we've retired as many in-flight operations as we have arguments for, move
             * forward in the AML stream.
             */
            let opcode = match context.opcode() {
                Ok(opcode) => opcode,
                Err(AmlError::RunOutOfStream) => {
                    /*
                     * We've reached the end of the current block. What we should do about this
                     * depends on what type of block it was.
                     */
                    // Hardening: a copy (the context is borrowed now).
                    match context.current_block.kind.clone() {
                        BlockKind::Table => {
                            break Ok(Object::Uninitialized.wrap());
                        }
                        BlockKind::Method { method_scope } => {
                            self.namespace.lock().remove_level(method_scope)?;

                            if let Some(prev_context) = self.pop_context(base) {
                                *context = prev_context;
                                continue;
                            } else {
                                /*
                                 * If there is no explicit `Return` op, the result is undefined. We
                                 * just return an uninitialized object.
                                 */
                                return Ok(Object::Uninitialized.wrap());
                            }
                        }
                        BlockKind::Scope { old_scope } => {
                            context.pop_block()?;
                            context.current_scope = old_scope;
                            // Go round the loop again to get the next opcode for the new block
                            continue;
                        }
                        BlockKind::Package => {
                            /*
                             * We've reached the end of the package. The in-flight op may have
                             * already been completed in the case of the package specifying all of
                             * its elements, or reach the end of the block here if it does not.
                             *
                             * In the latter case, fill in the rest of the package with
                             * *distinct* uninitialized objects, and go round again to complete the
                             * in-flight op.
                             */
                            // Hardening: was an `assert!`. If the last op in flight is not this package (an op inside it
                            // is missing operands), the package is malformed: going round again looped forever.
                            match context.in_flight.last_mut() {
                                Some(package_op) if package_op.op == Opcode::Package => {
                                    let num_elements_left =
                                        package_op.expected_arguments.saturating_sub(package_op.arguments.len());
                                    for _ in 0..num_elements_left {
                                        package_op.arguments.push(Argument::Object(Object::Uninitialized.wrap()));
                                    }
                                }
                                _ => return Err(AmlError::MalformedStream),
                            }

                            // XXX: don't remove the package's block. Refer to completion of
                            // package ops for rationale here.
                            continue;
                        }
                        BlockKind::VarPackage => {
                            // Hardening: the count was mis-computed (the op never completed, and the loop never ended) and
                            // could underflow; a missing count or a foreign op in flight is malformed.
                            match context.in_flight.last_mut() {
                                Some(package_op) if package_op.op == Opcode::VarPackage => {
                                    let Some(Argument::Object(total_elements)) = package_op.arguments.first()
                                    else {
                                        return Err(AmlError::MalformedStream);
                                    };
                                    let total_elements =
                                        total_elements.clone().unwrap_transparent_reference().as_integer()?;
                                    let total_elements = check_elements(&limits, total_elements)?;
                                    // The count, then exactly `total_elements` elements.
                                    package_op.arguments.truncate(total_elements + 1);
                                    while package_op.arguments.len() < total_elements + 1 {
                                        package_op.arguments.push(Argument::Object(Object::Uninitialized.wrap()));
                                    }
                                    package_op.expected_arguments = package_op.arguments.len();
                                }
                                _ => return Err(AmlError::MalformedStream),
                            }

                            // As above, leave the package's block.
                            continue;
                        }
                        BlockKind::IfThenBranch => {
                            context.pop_block()?;

                            /*
                             * Check for an else-branch, and skip over it. We need to handle the
                             * case here where there isn't a next byte - that just means the `If`
                             * is the last op in a block.
                             */
                            const DEF_ELSE_OP: u8 = 0xa1;
                            match context.peek() {
                                Ok(DEF_ELSE_OP) => {
                                    context.next()?;
                                    let start_pc = context.current_block.pc;
                                    let else_length = context.pkglength()?;
                                    let remaining = context.remaining(start_pc, else_length)?;
                                    context.skip(remaining)?;
                                }
                                Ok(_) => (),
                                Err(AmlError::RunOutOfStream) => (),
                                Err(other) => Err(other)?,
                            };

                            continue;
                        }
                        BlockKind::While { start_pc, iterations } => {
                            /*
                             * Go round again, and create a new in-flight op to have a look at the
                             * predicate.
                             */
                            // Hardening: bounded iterations per loop.
                            if iterations >= limits.max_loop_iterations {
                                return Err(AmlError::LimitExceeded(Limit::LoopIterations));
                            }
                            context.current_block.kind = BlockKind::While { start_pc, iterations: iterations + 1 };
                            context.current_block.pc = start_pc;
                            context.start(OpInFlight::new(Opcode::While, &[ResolveBehaviour::TermArg]));
                            continue;
                        }
                    }
                }
                Err(other_err) => return Err(other_err),
            };
            match opcode {
                Opcode::Zero => {
                    /*
                     * This represents a `Zero` operand that should create an `Integer` object in
                     * most places, but could also encode a `NullName` if we are expecting a
                     * `Target`.
                     */
                    if context.last_op()?.resolve_behaviour()? == ResolveBehaviour::Target {
                        context.last_op()?.arguments.push(Argument::Null);
                    } else {
                        context.last_op()?.arguments.push(Argument::Object(Object::Integer(0).wrap()));
                    }
                }
                Opcode::One => {
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(1).wrap()));
                }
                Opcode::Alias => {
                    let source = context.namestring()?;
                    let alias = context.namestring()?;

                    let mut namespace = self.namespace.lock();
                    let object = namespace.get(source.resolve(&context.current_scope)?)?.clone();
                    let alias = alias.resolve(&context.current_scope)?;
                    namespace.create_alias(alias, object)?;
                }
                Opcode::Name => {
                    let name = context.namestring()?;
                    context.start(OpInFlight::new_with(
                        Opcode::Name,
                        vec![Argument::Namestring(name)],
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                    ));
                }
                Opcode::BytePrefix => {
                    let value = context.next()?;
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(value as u64).wrap()));
                }
                Opcode::WordPrefix => {
                    let value = context.next_u16()?;
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(value as u64).wrap()));
                }
                Opcode::DWordPrefix => {
                    let value = context.next_u32()?;
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(value as u64).wrap()));
                }
                Opcode::StringPrefix => {
                    let str_start = context.current_block.pc;
                    while context.next()? != b'\0' {}
                    // Hardening: not UTF-8 is an error, not a panic.
                    let bytes = context
                        .current_block
                        .stream()
                        .get(str_start..context.current_block.pc.saturating_sub(1))
                        .ok_or(AmlError::MalformedStream)?;
                    self.check_size(bytes.len())?;
                    let str = String::from(str::from_utf8(bytes).map_err(|_| AmlError::MalformedStream)?);
                    context.last_op()?.arguments.push(Argument::Object(Object::String(str).wrap()));
                }
                Opcode::QWordPrefix => {
                    let value = context.next_u64()?;
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(value).wrap()));
                }
                Opcode::Scope => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;

                    let remaining_length = context.remaining(start_pc, pkg_length)?;

                    let new_scope = name.resolve(&context.current_scope)?;
                    self.namespace.lock().add_level(new_scope.clone(), NamespaceLevelKind::Scope)?;

                    let old_scope = mem::replace(&mut context.current_scope, new_scope);
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length, &limits)?;
                }
                Opcode::Buffer => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    context.start(OpInFlight::new_with(
                        Opcode::Buffer,
                        vec![Argument::TrackedPc(start_pc), Argument::PkgLength(pkg_length)],
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                    ));
                }
                Opcode::Package => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let num_elements = context.next()?;

                    let remaining_length = context.remaining(start_pc, pkg_length)?;
                    check_elements(&limits, num_elements as u64)?;

                    /*
                     * We now need to interpret an arbitrary number of package elements, bounded by
                     * the remaining pkglength. This may be less than `num_elements` - the
                     * remaining elements of the package are uninitialized. We utilise a
                     * combination of a block to manage the pkglength, plus an in-flight op to
                     * store interpreted arguments.
                     */
                    context.start(OpInFlight::new_dynamic(
                        Opcode::Package,
                        num_elements as usize,
                        &[ResolveBehaviour::AsPackageElements],
                    ));
                    context.start_new_block(BlockKind::Package, remaining_length, &limits)?;
                }
                Opcode::VarPackage => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let remaining_length = context.remaining(start_pc, pkg_length)?;

                    /*
                     * For variable packages, we're first going to parse a `TermArg` that encodes,
                     * dynamically, how many elements the package will have. We then accept as many
                     * elements as remain in the block, and we'll sort out how many are supposed to
                     * be in the package later.
                     */
                    context.start(OpInFlight::new_dynamic(
                        Opcode::VarPackage,
                        usize::MAX,
                        &[ResolveBehaviour::TermArg, ResolveBehaviour::AsPackageElements],
                    ));
                    context.start_new_block(BlockKind::VarPackage, remaining_length, &limits)?;
                }
                Opcode::Method => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let flags = MethodFlags(context.next()?);

                    let code_len = context.remaining(start_pc, pkg_length)?;
                    let code = context.take(code_len)?.to_vec();

                    let name = name.resolve(&context.current_scope)?;
                    self.namespace.lock().insert(name, Object::Method { code, flags }.wrap())?;
                }
                Opcode::External => {
                    let _name = context.namestring()?;
                    let _object_type = context.next()?;
                    let _arg_count = context.next()?;
                }
                Opcode::Mutex => {
                    let name = context.namestring()?;
                    let sync_level = context.next()?;

                    let name = name.resolve(&context.current_scope)?;
                    let mutex = self.handler.create_mutex();
                    self.namespace.lock().insert(name, Object::Mutex { mutex, sync_level }.wrap())?;
                }
                Opcode::Event => {
                    let name = context.namestring()?;

                    let name = name.resolve(&context.current_scope)?;
                    self.namespace.lock().insert(name, Object::Event(Arc::new(AtomicU64::new(0))).wrap())?;
                }
                // Hardening: were `todo!()`. Loading tables at run time is not supported: the method fails.
                Opcode::LoadTable => return Err(AmlError::Unsupported("LoadTable")),
                Opcode::Load => return Err(AmlError::Unsupported("Load")),
                Opcode::Stall => context.start(OpInFlight::new(Opcode::Stall, &[ResolveBehaviour::TermArg])),
                Opcode::Sleep => context.start(OpInFlight::new(Opcode::Sleep, &[ResolveBehaviour::TermArg])),
                Opcode::Acquire => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Release => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Signal => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Wait => context
                    .start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName, ResolveBehaviour::TermArg])),
                Opcode::Reset => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Notify => context
                    .start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName, ResolveBehaviour::TermArg])),
                Opcode::FromBCD | Opcode::ToBCD => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                Opcode::Revision => {
                    context.contribute_arg(Argument::Object(Object::Integer(INTERPRETER_REVISION).wrap()));
                }
                Opcode::Debug => {
                    context.contribute_arg(Argument::Object(Object::Debug.wrap()));
                }
                Opcode::Fatal => {
                    let typ = context.next()?;
                    let code = context.next_u32()?;
                    context.start(OpInFlight::new_with(
                        Opcode::Fatal,
                        vec![Argument::ByteData(typ), Argument::DWordData(code)],
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                    ));
                }
                Opcode::Timer => {
                    // Time has to be monotonically-increasing, in 100ns units
                    let time = self.handler.nanos_since_boot() / 100;
                    context.contribute_arg(Argument::Object(Object::Integer(time).wrap()));
                }
                Opcode::OpRegion => {
                    let name = context.namestring()?;
                    let region_space = context.next()?;
                    context.start(OpInFlight::new_with(
                        Opcode::OpRegion,
                        vec![Argument::Namestring(name), Argument::ByteData(region_space)],
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                        ],
                    ));
                }
                Opcode::DataRegion => {
                    let name = context.namestring()?;
                    context.start(OpInFlight::new_with(
                        Opcode::DataRegion,
                        vec![Argument::Namestring(name)],
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                        ],
                    ));
                }
                Opcode::Field => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let region_name = context.namestring()?;
                    let field_flags = context.next()?;

                    let region = self.namespace.lock().get(region_name.resolve(&context.current_scope)?)?.clone();
                    let kind = FieldUnitKind::Normal { region };
                    self.parse_field_list(context, kind, start_pc, pkg_length, field_flags)?;
                }
                Opcode::BankField => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let region_name = context.namestring()?;
                    let bank_name = context.namestring()?;

                    context.start(OpInFlight::new_with(
                        Opcode::BankField,
                        vec![
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(pkg_length),
                            Argument::Namestring(region_name),
                            Argument::Namestring(bank_name),
                        ],
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                        ],
                    ));
                }
                Opcode::IndexField => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let index_name = context.namestring()?;
                    let data_name = context.namestring()?;
                    let field_flags = context.next()?;

                    let (index, data) = {
                        let namespace = self.namespace.lock();
                        let (_, index) = namespace.search(&index_name, &context.current_scope)?;
                        let (_, data) = namespace.search(&data_name, &context.current_scope)?;
                        (index, data)
                    };
                    let kind = FieldUnitKind::Index { index, data };
                    self.parse_field_list(context, kind, start_pc, pkg_length, field_flags)?;
                }
                Opcode::Device | Opcode::ThermalZone => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;

                    let remaining_length = context.remaining(start_pc, pkg_length)?;

                    let new_scope = name.resolve(&context.current_scope)?;
                    let (kind, object) = if opcode == Opcode::Device {
                        (NamespaceLevelKind::Device, Object::Device)
                    } else {
                        (NamespaceLevelKind::ThermalZone, Object::ThermalZone)
                    };
                    let mut namespace = self.namespace.lock();
                    namespace.add_level(new_scope.clone(), kind)?;
                    namespace.insert(new_scope.clone(), object.wrap())?;

                    let old_scope = mem::replace(&mut context.current_scope, new_scope);
                    drop(namespace);
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length, &limits)?;
                }
                Opcode::Processor => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let proc_id = context.next()?;
                    let pblk_address = context.next_u32()?;
                    let pblk_length = context.next()?;

                    let remaining_length = context.remaining(start_pc, pkg_length)?;

                    let new_scope = name.resolve(&context.current_scope)?;
                    let object = Object::Processor { proc_id, pblk_address, pblk_length };
                    let mut namespace = self.namespace.lock();
                    namespace.add_level(new_scope.clone(), NamespaceLevelKind::Processor)?;
                    namespace.insert(new_scope.clone(), object.wrap())?;

                    let old_scope = mem::replace(&mut context.current_scope, new_scope);
                    drop(namespace);
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length, &limits)?;
                }
                Opcode::PowerRes => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let system_level = context.next()?;
                    let resource_order = context.next_u16()?;

                    let remaining_length = context.remaining(start_pc, pkg_length)?;

                    let new_scope = name.resolve(&context.current_scope)?;
                    let object = Object::PowerResource { system_level, resource_order };
                    let mut namespace = self.namespace.lock();
                    namespace.add_level(new_scope.clone(), NamespaceLevelKind::PowerResource)?;
                    namespace.insert(new_scope.clone(), object.wrap())?;

                    let old_scope = mem::replace(&mut context.current_scope, new_scope);
                    drop(namespace);
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length, &limits)?;
                }
                Opcode::Local(local) => {
                    let local = context.locals[local as usize].clone();
                    context.last_op()?.arguments.push(Argument::Object(
                        Object::Reference { kind: ReferenceKind::LocalOrArg, inner: local }.wrap(),
                    ));
                }
                Opcode::Arg(arg) => {
                    let arg = context.args[arg as usize].clone();
                    context.last_op()?.arguments.push(Argument::Object(
                        Object::Reference { kind: ReferenceKind::LocalOrArg, inner: arg }.wrap(),
                    ));
                }
                Opcode::Store => context.start(OpInFlight::new(
                    Opcode::Store,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::SuperName],
                )),
                Opcode::RefOf => context.start(OpInFlight::new(Opcode::RefOf, &[ResolveBehaviour::SuperName])),
                Opcode::CondRefOf => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::SuperNameIfExists, ResolveBehaviour::Target],
                )),

                Opcode::DualNamePrefix
                | Opcode::MultiNamePrefix
                | Opcode::Digit(_)
                | Opcode::NameChar(_)
                | Opcode::RootChar
                | Opcode::ParentPrefixChar => {
                    context.current_block.pc -= 1;
                    let name = context.namestring()?;

                    let behaviour = match context.in_flight.last() {
                        Some(op) => op.resolve_behaviour()?,
                        None => ResolveBehaviour::TermArg,
                    };
                    match behaviour {
                        // XXX: `NullName` is handled separately given its ambiguity with `Zero`
                        ResolveBehaviour::SuperName | ResolveBehaviour::Target => {
                            let object = self.namespace.lock().search(&name, &context.current_scope);
                            match object {
                                Ok((resolved_name, object)) => {
                                    let op = context.last_op()?;
                                    // Hardening: `Notify` hands the host the object's resolved path.
                                    if op.op == Opcode::Notify && op.arguments.is_empty() {
                                        op.target_name = Some(resolved_name);
                                    }
                                    op.arguments.push(Argument::Object(object));
                                }
                                Err(err) => Err(err)?,
                            }
                        }
                        ResolveBehaviour::SuperNameIfExists => {
                            let object = self.namespace.lock().search(&name, &context.current_scope);
                            match object {
                                Ok((_resolved_name, object)) => {
                                    context.last_op()?.arguments.push(Argument::Object(object));
                                }
                                Err(AmlError::ObjectDoesNotExist(_)) => {
                                    let reference = Object::Reference {
                                        kind: ReferenceKind::Unresolved,
                                        inner: Object::String(name.to_string()).wrap(),
                                    };
                                    context.last_op()?.arguments.push(Argument::Object(reference.wrap()));
                                }
                                Err(err) => Err(err)?,
                            }
                        }
                        ResolveBehaviour::TermArg => {
                            let object = self.namespace.lock().search(&name, &context.current_scope);
                            match object {
                                Ok((resolved_name, object)) => {
                                    if let Object::Method { flags, .. } | Object::NativeMethod { flags, .. } =
                                        *object
                                    {
                                        context.start(OpInFlight::new_with_dynamic(
                                            Opcode::InternalMethodCall,
                                            vec![Argument::Object(object), Argument::Namestring(resolved_name)],
                                            flags.arg_count(),
                                            &[
                                                ResolveBehaviour::Placeholder,
                                                ResolveBehaviour::Placeholder,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                            ],
                                        ))
                                    } else if let Object::FieldUnit(ref field) = *object {
                                        let value = self.do_field_read(field)?;
                                        context.last_op()?.arguments.push(Argument::Object(value));
                                    } else {
                                        context.last_op()?.arguments.push(Argument::Object(object));
                                    }
                                }
                                Err(err) => Err(err)?,
                            }
                        }
                        ResolveBehaviour::AsPackageElements => {
                            context
                                .last_op()?
                                .arguments
                                .push(Argument::Object(Object::String(name.to_string()).wrap()));
                        }
                        // Hardening: was `panic!`.
                        ResolveBehaviour::Placeholder => return Err(AmlError::MalformedStream),
                    }
                }

                Opcode::Add
                | Opcode::Subtract
                | Opcode::Multiply
                | Opcode::ShiftLeft
                | Opcode::ShiftRight
                | Opcode::Mod
                | Opcode::Nand
                | Opcode::And
                | Opcode::Or
                | Opcode::Nor
                | Opcode::Xor
                | Opcode::Concat => {
                    context.start(OpInFlight::new(
                        opcode,
                        &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                    ));
                }

                Opcode::Divide => context.start(OpInFlight::new(
                    Opcode::Divide,
                    &[
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::Target,
                        ResolveBehaviour::Target,
                    ],
                )),
                Opcode::Increment | Opcode::Decrement => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName]))
                }
                Opcode::Not => context
                    .start(OpInFlight::new(Opcode::Not, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target])),
                Opcode::FindSetLeftBit | Opcode::FindSetRightBit => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                Opcode::DerefOf => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg])),
                Opcode::ConcatRes => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),
                Opcode::SizeOf => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Index => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),
                /*
                 * TODO
                 * Match is a difficult opcode to parse, as it interleaves dynamic arguments and
                 * random bytes that need to be extracted as you go. I think we'll need to use 1+
                 * internal in-flight ops to parse the static bytedatas as we go, and then retire
                 * the real op at the end.
                 */
                // Hardening: was `todo!()`.
                Opcode::Match => return Err(AmlError::Unsupported("Match")),

                Opcode::CreateBitField
                | Opcode::CreateByteField
                | Opcode::CreateWordField
                | Opcode::CreateDWordField
                | Opcode::CreateQWordField => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg; 2]))
                }
                Opcode::CreateField => {
                    context.start(OpInFlight::new(Opcode::CreateField, &[ResolveBehaviour::TermArg; 3]))
                }

                Opcode::LAnd
                | Opcode::LOr
                | Opcode::LNot
                | Opcode::LNotEqual
                | Opcode::LLessEqual
                | Opcode::LGreaterEqual
                | Opcode::LEqual
                | Opcode::LGreater
                | Opcode::LLess => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg; 2]));
                }

                Opcode::ToBuffer | Opcode::ToDecimalString | Opcode::ToHexString | Opcode::ToInteger => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                Opcode::ToString => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),

                Opcode::ObjectType => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                // Hardening: was `todo!()`; done as a store (see its completion).
                Opcode::CopyObject => context.start(OpInFlight::new(
                    Opcode::CopyObject,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::SuperName],
                )),
                Opcode::Mid => context.start(OpInFlight::new(
                    Opcode::Mid,
                    &[
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::Target,
                    ],
                )),
                Opcode::If => {
                    let start_pc = context.current_block.pc;
                    let then_length = context.pkglength()?;
                    context.start(OpInFlight::new_with(
                        Opcode::If,
                        vec![Argument::TrackedPc(start_pc), Argument::PkgLength(then_length)],
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                    ));
                }
                Opcode::Else => return Err(AmlError::ElseFoundWithoutCorrespondingIf),
                Opcode::While => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let remaining_length = context.remaining(start_pc, pkg_length)?;
                    context.start_new_block(
                        BlockKind::While { start_pc: context.current_block.pc, iterations: 0 },
                        remaining_length,
                        &limits,
                    )?;
                    context.start(OpInFlight::new(Opcode::While, &[ResolveBehaviour::TermArg]));
                }
                Opcode::Continue => {
                    // Hardening: the loop's own block becomes current again (the inner block's pc was moved instead),
                    // and a `Continue` counts as an iteration.
                    loop {
                        if let BlockKind::While { start_pc, iterations } = context.current_block.kind {
                            if iterations >= limits.max_loop_iterations {
                                return Err(AmlError::LimitExceeded(Limit::LoopIterations));
                            }
                            context.current_block.kind = BlockKind::While { start_pc, iterations: iterations + 1 };
                            context.current_block.pc = start_pc;
                            break;
                        }
                        if context.block_stack.is_empty() {
                            Err(AmlError::ContinueOutsideOfWhile)?
                        }
                        context.pop_block()?;
                    }
                    context.start(OpInFlight::new(Opcode::While, &[ResolveBehaviour::TermArg]));
                }
                Opcode::Break => {
                    // Hardening: were `unwrap`s.
                    loop {
                        let in_while = matches!(context.current_block.kind, BlockKind::While { .. });
                        if !in_while && context.block_stack.is_empty() {
                            Err(AmlError::BreakOutsideOfWhile)?
                        }
                        context.pop_block()?;
                        if in_while {
                            break;
                        }
                    }
                }
                Opcode::Return => context.start(OpInFlight::new(Opcode::Return, &[ResolveBehaviour::TermArg])),
                Opcode::Noop => {}
                Opcode::Breakpoint => {
                    self.handler.breakpoint();
                }
                Opcode::Ones => {
                    context.last_op()?.arguments.push(Argument::Object(Object::Integer(u64::MAX).wrap()));
                }

                // Hardening: was `panic!()`; never decoded from a stream.
                Opcode::InternalMethodCall => return Err(AmlError::MalformedStream),
            }
        }
    }

    fn parse_field_list(
        &self,
        context: &mut MethodContext,
        kind: FieldUnitKind,
        start_pc: usize,
        pkg_length: usize,
        flags: u8,
    ) -> Result<(), AmlError> {
        const RESERVED_FIELD: u8 = 0x00;
        const ACCESS_FIELD: u8 = 0x01;
        const CONNECT_FIELD: u8 = 0x02;
        const EXTENDED_ACCESS_FIELD: u8 = 0x03;

        let mut field_offset: usize = 0;
        // Hardening: `AccessAs` changes the access type of the fields after it.
        let mut flags = flags;
        let end = start_pc.saturating_add(pkg_length);

        while context.current_block.pc < end {
            match context.next()? {
                RESERVED_FIELD => {
                    let length = context.pkglength()?;
                    field_offset = field_offset.checked_add(length).ok_or(AmlError::MalformedStream)?;
                }
                ACCESS_FIELD => {
                    /*
                     * These aren't actually fields themselves, but are created by `AccessAs` AML
                     * elements. They change the access type and attributes for remaining fields in
                     * the list.
                     */
                    // Hardening: was `todo!()`. The access type replaces the low nibble of the flags; the attribute (an
                    // SMBus or GenericSerialBus protocol) is not used.
                    let access_type = context.next()?;
                    let _access_attrib = context.next()?;
                    flags = (flags & 0xf0) | (access_type & 0x0f);
                }
                CONNECT_FIELD => {
                    // Hardening: was `todo!()`. `Connection(...)` names the GPIO or serial bus resource of the fields
                    // after it: a NameString or a buffer. It is consumed; such fields need a handler for their
                    // space (GeneralPurposeIo or GenericSerialBus), which is given the region only.
                    match context.peek()? {
                        0x11 => {
                            context.next()?;
                            let start = context.current_block.pc;
                            let length = context.pkglength()?;
                            let remaining = context.remaining(start, length)?;
                            context.skip(remaining)?;
                        }
                        _ => {
                            context.namestring()?;
                        }
                    }
                }
                EXTENDED_ACCESS_FIELD => {
                    // Hardening: was `todo!()`. Access type, attribute and length: the type is used as for `AccessAs`.
                    let access_type = context.next()?;
                    let _access_attrib = context.next()?;
                    let _access_length = context.next()?;
                    flags = (flags & 0xf0) | (access_type & 0x0f);
                }
                _ => {
                    context.current_block.pc -= 1;
                    // TODO: this should only ever be a nameseg really...
                    let field_name = context.namestring()?;
                    let field_length = context.pkglength()?;

                    let field = Object::FieldUnit(FieldUnit {
                        kind: kind.clone(),
                        bit_index: field_offset,
                        bit_length: field_length,
                        flags: FieldFlags(flags),
                    });
                    self.namespace.lock().insert(field_name.resolve(&context.current_scope)?, field.wrap())?;

                    field_offset = field_offset.checked_add(field_length).ok_or(AmlError::MalformedStream)?;
                }
            }
        }

        Ok(())
    }

    fn do_binary_maths(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(left), Argument::Object(right), target, ..] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let target2 = if op.op == Opcode::Divide { op.arguments.get(3) } else { None };

        let left = left.clone().unwrap_transparent_reference().as_integer()?;
        let right = right.clone().unwrap_transparent_reference().as_integer()?;

        let result = match op.op {
            Opcode::Add => left.wrapping_add(right),
            Opcode::Subtract => left.wrapping_sub(right),
            Opcode::Multiply => left.wrapping_mul(right),
            // Hardening: a zero divisor was a panic.
            Opcode::Divide => {
                if right == 0 {
                    return Err(AmlError::DivideByZero);
                }
                if let Some(remainder) = target2 {
                    self.do_store(remainder, Object::Integer(left.wrapping_rem(right)).wrap())?;
                }
                left.wrapping_div_euclid(right)
            }
            // Hardening: a shift of 64 or more is zero (it used to wrap the count).
            Opcode::ShiftLeft => u32::try_from(right).ok().and_then(|r| left.checked_shl(r)).unwrap_or(0),
            Opcode::ShiftRight => u32::try_from(right).ok().and_then(|r| left.checked_shr(r)).unwrap_or(0),
            Opcode::Mod => left.checked_rem(right).ok_or(AmlError::DivideByZero)?,
            Opcode::Nand => !(left & right),
            Opcode::And => left & right,
            Opcode::Or => left | right,
            Opcode::Nor => !(left | right),
            Opcode::Xor => left ^ right,
            _ => return Err(AmlError::MalformedStream),
        };

        let result = Object::Integer(result).wrap();
        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_unary_maths(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        // Hardening: `Not` and the bit finders take a target too; only the operand is used here.
        let Some(Argument::Object(operand)) = op.arguments.first() else {
            return Err(AmlError::MalformedStream);
        };
        let operand = operand.clone().unwrap_transparent_reference().as_integer()?;

        let result = match op.op {
            Opcode::FindSetLeftBit => {
                if operand == 0 {
                    0
                } else {
                    /*
                     * This is a particularly important place to respect the integer width as set
                     * by the DSDT revision.
                     */
                    if self.dsdt_revision >= 2 {
                        (operand.leading_zeros() + 1) as u64
                    } else {
                        ((operand as u32).leading_zeros() + 1) as u64
                    }
                }
            }
            Opcode::FindSetRightBit => {
                if operand == 0 {
                    0
                } else {
                    (operand.trailing_zeros() + 1) as u64
                }
            }
            Opcode::Not => {
                if operand == 0 {
                    u64::MAX
                } else {
                    0
                }
            }
            _ => return Err(AmlError::MalformedStream),
        };

        // Hardening: the target was never stored to (and the pattern with it present always failed).
        let result = match op.arguments.get(1) {
            Some(target) => self.do_store(target, Object::Integer(result).wrap())?,
            None => Object::Integer(result).wrap(),
        };
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_logical_op(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        if op.op == Opcode::LNot {
            let [Argument::Object(operand)] = &op.arguments[..] else {
                return Err(AmlError::MalformedStream);
            };
            let operand = operand.clone().unwrap_transparent_reference().as_integer()?;
            let result = if operand == 0 { u64::MAX } else { 0 };

            context.contribute_arg(Argument::Object(Object::Integer(result).wrap()));
            context.retire_op(op);
            return Ok(());
        }

        let [Argument::Object(left), Argument::Object(right)] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let left = left.clone().unwrap_transparent_reference();
        let right = right.clone().unwrap_transparent_reference();

        /*
         * Some of these operations allow strings and buffers to be used as operands. Apparently
         * NT's interpreter just takes the first 4 bytes of the string/buffer and casts them as an
         * integer...
         */
        let (left, right) = match *left {
            Object::Integer(left) => (left, right.as_integer()?),
            Object::String(ref left) => {
                let left = {
                    let mut bytes = [0u8; 4];
                    let left_bytes = left.as_bytes();
                    let bytes_to_use = usize::min(4, left_bytes.len());
                    (bytes[0..bytes_to_use]).copy_from_slice(&left_bytes[0..bytes_to_use]);
                    u32::from_le_bytes(bytes) as u64
                };
                let right = {
                    let mut bytes = [0u8; 4];
                    let right = right.as_string()?;
                    let right_bytes = right.as_bytes();
                    let bytes_to_use = usize::min(4, right_bytes.len());
                    (bytes[0..bytes_to_use]).copy_from_slice(&right_bytes[0..bytes_to_use]);
                    u32::from_le_bytes(bytes) as u64
                };
                (left, right)
            }
            Object::Buffer(ref left) => {
                // Hardening: a buffer longer than four bytes, or a right side that is no buffer, were panics.
                let right = right.to_buffer(if self.dsdt_revision >= 2 { 8 } else { 4 })?;
                let first4 = |bytes: &[u8]| {
                    let mut out = [0u8; 4];
                    let n = usize::min(4, bytes.len());
                    out[..n].copy_from_slice(&bytes[..n]);
                    u32::from_le_bytes(out) as u64
                };
                (first4(left), first4(&right))
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::LogicalOp, typ: left.typ() })?,
        };

        let result = match op.op {
            Opcode::LAnd => (left > 0) && (right > 0),
            Opcode::LOr => (left > 0) || (right > 0),
            Opcode::LNotEqual => left != right,
            Opcode::LLessEqual => left <= right,
            Opcode::LGreaterEqual => left >= right,
            Opcode::LEqual => left == right,
            Opcode::LGreater => left > right,
            Opcode::LLess => left < right,
            _ => return Err(AmlError::MalformedStream),
        };
        let result = if result { Object::Integer(u64::MAX) } else { Object::Integer(0) };

        context.contribute_arg(Argument::Object(result.wrap()));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_buffer(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(operand), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let operand = operand.clone().unwrap_transparent_reference();

        let result = match *operand {
            Object::Buffer(ref bytes) => Object::Buffer(bytes.clone()),
            Object::Integer(value) => {
                if self.dsdt_revision >= 2 {
                    Object::Buffer(value.to_le_bytes().to_vec())
                } else {
                    Object::Buffer((value as u32).to_le_bytes().to_vec())
                }
            }
            Object::String(ref value) => {
                // XXX: an empty string is converted to an empty buffer, *without* the null-terminator
                if value.is_empty() {
                    Object::Buffer(vec![])
                } else {
                    let mut bytes = value.as_bytes().to_vec();
                    bytes.push(b'\0');
                    Object::Buffer(bytes)
                }
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToBuffer, typ: operand.typ() })?,
        }
        .wrap();

        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_integer(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(operand), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let operand = operand.clone().unwrap_transparent_reference();

        let result = match *operand {
            Object::Integer(value) => Object::Integer(value),
            Object::Buffer(ref bytes) => {
                /*
                 * The spec says this should respect the revision of the current definition block.
                 * Apparently, the NT interpreter always uses the first 8 bytes of the buffer.
                 */
                let mut to_interpret = [0u8; 8];
                (to_interpret[0..usize::min(bytes.len(), 8)]).copy_from_slice(bytes);
                Object::Integer(u64::from_le_bytes(to_interpret))
            }
            Object::String(ref value) => {
                /*
                 * This is about the same level of effort as ACPICA puts in. The uACPI test suite
                 * has tests that this fails - namely because of support for octal, signs, strings
                 * that won't fit in a `u64` etc. We probably need to write a more robust parser
                 * 'real' parser to handle those cases.
                 */
                if let Some(value) = value.strip_prefix("0x") {
                    let parsed = u64::from_str_radix(value, 16).map_err(|_| {
                        AmlError::InvalidOperationOnObject { op: Operation::ToInteger, typ: ObjectType::String }
                    })?;
                    Object::Integer(parsed)
                } else {
                    let parsed = str::parse::<u64>(value).map_err(|_| AmlError::InvalidOperationOnObject {
                        op: Operation::ToInteger,
                        typ: ObjectType::String,
                    })?;
                    Object::Integer(parsed)
                }
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToBuffer, typ: operand.typ() })?,
        }
        .wrap();

        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_string(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(source), Argument::Object(length), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let source = source.clone().unwrap_transparent_reference();
        let source = source.as_buffer()?;
        let length =
            usize::try_from(length.clone().unwrap_transparent_reference().as_integer()?).unwrap_or(usize::MAX);

        // Hardening: up to the first NUL, not including it (it was included), with no `unwrap`.
        let bytes = source.split(|b| *b == b'\0').next().unwrap_or(&[]);
        let bytes = &bytes[..usize::min(length, bytes.len())];
        let string = str::from_utf8(bytes).map_err(|_| AmlError::InvalidOperationOnObject {
            op: Operation::ToString,
            typ: ObjectType::Buffer,
        })?;
        let result = Object::String(string.to_string()).wrap();

        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    /// Perform a `ToDecimalString` or `ToHexString` operation
    fn do_to_dec_hex_string(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(operand), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let operand = operand.clone().unwrap_transparent_reference();
        let decimal = op.op == Opcode::ToDecimalString;

        let result = match *operand {
            Object::String(ref value) => Object::String(value.clone()),
            Object::Integer(value) => {
                if decimal {
                    Object::String(value.to_string())
                } else {
                    Object::String(alloc::format!("{value:#x}"))
                }
            }
            Object::Buffer(ref bytes) => {
                // Hardening: each byte becomes up to five characters: bounded before it is built.
                self.check_size(bytes.len().saturating_mul(5))?;
                let mut string = String::new();
                for byte in bytes {
                    if decimal {
                        string.push_str(&alloc::format!("{byte},"));
                    } else {
                        string.push_str(&alloc::format!("{byte:#04X},"));
                    }
                }
                // Remove last comma, if present
                string.pop();
                Object::String(string)
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToDecOrHexString, typ: operand.typ() })?,
        }
        .wrap();

        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_mid(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(source), Argument::Object(index), Argument::Object(length), target] =
            &op.arguments[..]
        else {
            return Err(AmlError::MalformedStream);
        };
        let index =
            usize::try_from(index.clone().unwrap_transparent_reference().as_integer()?).unwrap_or(usize::MAX);
        let length =
            usize::try_from(length.clone().unwrap_transparent_reference().as_integer()?).unwrap_or(usize::MAX);
        // Hardening: the end was `min(index + length, index + len)` (past the end, and an overflow); it is `len` at most,
        // and a string is cut on bytes, never inside a character.
        let range = |len: usize| index.min(len)..index.saturating_add(length).min(len);

        let source = source.clone().unwrap_transparent_reference();
        let result = match *source {
            Object::String(ref string) => {
                let bytes = &string.as_bytes()[range(string.len())];
                Object::String(String::from_utf8_lossy(bytes).into_owned())
            }
            Object::Buffer(ref buffer) => Object::Buffer(buffer[range(buffer.len())].to_vec()),
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::Mid, typ: source.typ() })?,
        }
        .wrap();

        self.do_store(target, result.clone())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_concat(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(source1), Argument::Object(source2), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let source1 = source1.clone().unwrap_transparent_reference();
        let source2 = source2.clone().unwrap_transparent_reference();

        fn resolve_as_string(obj: &Object) -> String {
            match obj {
                Object::Uninitialized => "[Uninitialized Object]".to_string(),
                Object::Buffer(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                Object::BufferField { .. } => "[Buffer Field]".to_string(),
                Object::Device => "[Device]".to_string(),
                Object::Event(_) => "[Event]".to_string(),
                Object::FieldUnit(_) => "[Field]".to_string(),
                Object::Integer(value) => value.to_string(),
                Object::Method { .. } | Object::NativeMethod { .. } => "[Control Method]".to_string(),
                Object::Mutex { .. } => "[Mutex]".to_string(),
                // Hardening: no recursion through references (a cycle recursed forever).
                Object::Reference { .. } => "[Reference]".to_string(),
                Object::OpRegion(_) => "[Operation Region]".to_string(),
                Object::Package(_) => "[Package]".to_string(),
                Object::PowerResource { .. } => "[Power Resource]".to_string(),
                Object::Processor { .. } => "[Processor]".to_string(),
                // Hardening: was `todo!()`.
                Object::RawDataBuffer => "[Raw Data Buffer]".to_string(),
                Object::String(value) => value.clone(),
                Object::ThermalZone => "[Thermal Zone]".to_string(),
                Object::Debug => "[Debug Object]".to_string(),
            }
        }

        let width = if self.dsdt_revision >= 2 { 8 } else { 4 };
        let result = match source1.typ() {
            ObjectType::Integer => {
                let source1 = source1.as_integer()?;
                let source2 = source2.to_integer(width)?;
                let mut buffer = Vec::new();
                if self.dsdt_revision >= 2 {
                    buffer.extend_from_slice(&source1.to_le_bytes());
                    buffer.extend_from_slice(&source2.to_le_bytes());
                } else {
                    buffer.extend_from_slice(&(source1 as u32).to_le_bytes());
                    buffer.extend_from_slice(&(source2 as u32).to_le_bytes());
                }
                Object::Buffer(buffer).wrap()
            }
            ObjectType::Buffer => {
                let first = source1.as_buffer()?;
                let second = source2.to_buffer(width)?;
                // Hardening: a loop of `Concat`s doubled a buffer without bound.
                self.check_size(first.len().saturating_add(second.len()))?;
                let mut buffer = first.to_vec();
                buffer.extend(second);
                Object::Buffer(buffer).wrap()
            }
            _ => {
                let source1 = resolve_as_string(&source1);
                let source2 = resolve_as_string(&source2);
                self.check_size(source1.len().saturating_add(source2.len()))?;
                Object::String(source1 + &source2).wrap()
            }
        };

        let result = self.do_store(target, result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_from_bcd(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        // Hardening: the target was never accepted (the pattern always failed with the target present) or stored to.
        let [Argument::Object(value), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let mut value = value.clone().unwrap_transparent_reference().as_integer()?;

        let mut result: u64 = 0;
        let mut i: u64 = 1;
        while value > 0 {
            result = result.wrapping_add((value & 0x0f).wrapping_mul(i));
            i = i.wrapping_mul(10);
            value >>= 4;
        }

        let result = self.do_store(target, Object::Integer(result).wrap())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_bcd(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(value), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let mut value = value.clone().unwrap_transparent_reference().as_integer()?;

        // Hardening: a shift of 64 or more (a value of 17 digits or more) was an overflow.
        let mut result: u64 = 0;
        let mut i: u32 = 0;
        while value > 0 && i < 16 {
            result |= (value % 10) << (4 * i);
            value /= 10;
            i += 1;
        }

        let result = self.do_store(target, Object::Integer(result).wrap())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_size_of(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(object)] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let object = object.clone().unwrap_reference();

        let result = match *object {
            Object::Buffer(ref buffer) => buffer.len(),
            Object::String(ref str) => str.len(),
            Object::Package(ref package) => package.len(),
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::SizeOf, typ: object.typ() })?,
        };

        context.contribute_arg(Argument::Object(Object::Integer(result as u64).wrap()));
        context.retire_op(op);
        Ok(())
    }

    fn do_index(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        let [Argument::Object(object), Argument::Object(index_value), target] = &op.arguments[..] else {
            return Err(AmlError::MalformedStream);
        };
        let object = object.clone().unwrap_transparent_reference();
        let index_value = usize::try_from(index_value.clone().unwrap_transparent_reference().as_integer()?)
            .map_err(|_| AmlError::IndexOutOfBounds)?;

        let byte_field = |len: usize| -> Result<Object, AmlError> {
            if index_value >= len {
                Err(AmlError::IndexOutOfBounds)?
            }
            Ok(Object::Reference {
                kind: ReferenceKind::RefOf,
                inner: Object::BufferField { buffer: object.clone(), offset: index_value * 8, length: 8 }.wrap(),
            })
        };
        let result = match *object {
            Object::Buffer(ref buffer) => byte_field(buffer.len())?,
            Object::String(ref string) => byte_field(string.len())?,
            Object::Package(ref package) => {
                let Some(element) = package.get(index_value) else { Err(AmlError::IndexOutOfBounds)? };
                Object::Reference { kind: ReferenceKind::RefOf, inner: element.clone() }
            }
            _ => Err(AmlError::IndexOutOfBounds)?,
        }
        .wrap();

        self.do_store(target, result.clone())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    /// Hardening: rewritten without panics or `todo!`s, and without holding a mutable reference to the target while the
    /// source (which may be the same object) is read: the source's value is taken first.
    ///
    /// - A named integer takes the source converted to an integer; a buffer field or field unit is written.
    /// - A reference made by `Index` or `RefOf` (the usual way to fill a package's element in `_BST` or `_BIX`)
    ///   stores into what it refers to, by the same rules.
    /// - A local or an argument, and a named `Buffer`, `String`, `Package` or uninitialized object, take the
    ///   source's value (the implicit conversions of ACPI 19.3.5 are not done; this is what NT is said to do).
    /// - Anything else (a device, a method, a mutex...) is an error.
    fn do_store(&self, target: &Argument, object: WrappedObject) -> Result<WrappedObject, AmlError> {
        let object = object.unwrap_transparent_reference();
        let to_return = object.clone();
        match target {
            Argument::Null => Ok(to_return),
            Argument::Object(target) => {
                self.store_into(target, &object, 0)?;
                Ok(to_return)
            }
            Argument::Namestring(_)
            | Argument::ByteData(_)
            | Argument::DWordData(_)
            | Argument::TrackedPc(_)
            | Argument::PkgLength(_) => Err(AmlError::MalformedStream),
        }
    }

    /// The value a source object stores: a field unit or buffer field is read first.
    fn store_value(&self, object: &WrappedObject) -> Result<Object, AmlError> {
        match **object {
            Object::FieldUnit(ref field) => Ok((*self.do_field_read(field)?).clone()),
            Object::BufferField { ref length, .. } => {
                // Hardening: checked before it is allocated (the allocation came first).
                let bytes = length.div_ceil(8).max(8);
                self.check_size(bytes)?;
                let mut buffer = vec![0u8; bytes];
                object.read_buffer_field(&mut buffer)?;
                if *length <= 64 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&buffer[..8]);
                    Ok(Object::Integer(u64::from_le_bytes(bytes)))
                } else {
                    buffer.truncate(length.div_ceil(8));
                    Ok(Object::Buffer(buffer))
                }
            }
            ref other => Ok(other.clone()),
        }
    }

    fn store_into(&self, target: &WrappedObject, object: &WrappedObject, depth: usize) -> Result<(), AmlError> {
        // A chain of references is followed a few steps at most.
        if depth > 8 {
            return Err(AmlError::InvalidOperationOnObject { op: Operation::Store, typ: ObjectType::Reference });
        }
        let width = if self.dsdt_revision >= 2 { 8 } else { 4 };
        match **target {
            Object::FieldUnit(ref field) => {
                let value = self.store_value(object)?.wrap();
                return self.do_field_write(field, value);
            }
            Object::BufferField { .. } => {
                let bytes = self.store_value(object)?.to_buffer(width)?;
                let token = self.object_token.lock();
                // SAFETY: the object token is held; `bytes` is an owned copy of the source, so nothing borrows the
                // field, and the field's buffer is a different object from the field itself.
                return unsafe { target.gain_mut(&token) }.write_buffer_field(&bytes, &token);
            }
            Object::Reference { kind, ref inner } => {
                return match kind {
                    ReferenceKind::RefOf => self.store_into(inner, object, depth + 1),
                    ReferenceKind::LocalOrArg => {
                        // A local or argument that holds a reference stores through it; otherwise the local's
                        // value is replaced.
                        if let Object::Reference { kind: ReferenceKind::RefOf, .. } = **inner {
                            self.store_into(inner, object, depth + 1)
                        } else {
                            let value = self.store_value(object)?;
                            let token = self.object_token.lock();
                            // SAFETY: the object token is held, and `value` is owned (read out of the source
                            // before), so no reference into the local's slot is alive while it is replaced.
                            unsafe { *inner.gain_mut(&token) = value };
                            Ok(())
                        }
                    }
                    // Hardening: was `todo!()`.
                    ReferenceKind::Unresolved => Err(AmlError::InvalidOperationOnObject {
                        op: Operation::Store,
                        typ: ObjectType::Reference,
                    }),
                };
            }
            Object::Debug => {
                self.handler.handle_debug(object);
                return Ok(());
            }
            _ => {}
        }

        let value = self.store_value(object)?;
        let token = self.object_token.lock();
        // SAFETY: the object token is held, and `value` is owned (read out of the source before the token was
        // taken), so the only reference into `target` is this one, for the length of the match.
        match unsafe { target.gain_mut(&token) } {
            Object::Integer(target) => {
                *target = value.to_integer(width)?;
                Ok(())
            }
            slot @ (Object::Uninitialized | Object::Buffer(_) | Object::String(_) | Object::Package(_)) => {
                *slot = value;
                Ok(())
            }
            // Hardening: was `panic!("Stores to objects like {:?} are not yet supported")`.
            other => Err(AmlError::InvalidOperationOnObject { op: Operation::Store, typ: other.typ() }),
        }
    }

    /// Do a read from a field by performing one or more well-formed accesses to the underlying
    /// operation regions, and then shifting and masking the resulting value as appropriate. Will
    /// return either an `Integer` or `Buffer` as appropriate, guided by the size of the field
    /// and expected integer size (as per the DSDT revision).
    fn do_field_read(&self, field: &FieldUnit) -> Result<WrappedObject, AmlError> {
        self.field_read(field, 0)
    }

    fn field_read(&self, field: &FieldUnit, depth: usize) -> Result<WrappedObject, AmlError> {
        let needs_buffer = if self.dsdt_revision >= 2 { field.bit_length > 64 } else { field.bit_length > 32 };
        let access_width_bits = field.flags.access_type_bytes()? * 8;

        trace!("AML field read. Field = {:?}", field);

        // Hardening: the buffer is the field's length in bytes (it was its length in bits), checked first.
        let byte_length = field.bit_length.div_ceil(8);
        self.check_size(byte_length)?;
        let mut output = vec![0u8; usize::max(byte_length, 8)];

        /*
         * Break the field read into native reads that respect the region's access width.
         * Copy each potentially-unaligned part into the destination's bit range.
         */
        let mut read_so_far = 0;
        let mut i = 0;
        while read_so_far < field.bit_length {
            let bit =
                field.bit_index.checked_add(i * access_width_bits).ok_or(AmlError::RegionAccessOutOfBounds)?;
            let aligned_offset = object::align_down(bit, access_width_bits);
            let raw = self.field_access(field, aligned_offset / 8, access_width_bits / 8, None, depth)?;
            let src_index = if i == 0 { field.bit_index % access_width_bits } else { 0 };
            let length = usize::min(field.bit_length - read_so_far, access_width_bits - src_index);

            object::copy_bits(&raw.to_le_bytes(), src_index, &mut output, read_so_far, length);
            read_so_far += length;
            i += 1;
        }

        if needs_buffer {
            output.truncate(byte_length);
            Ok(Object::Buffer(output).wrap())
        } else {
            let mut value = [0u8; 8];
            value.copy_from_slice(&output[..8]);
            Ok(Object::Integer(u64::from_le_bytes(value)).wrap())
        }
    }

    fn do_field_write(&self, field: &FieldUnit, value: WrappedObject) -> Result<(), AmlError> {
        self.field_write(field, value, 0)
    }

    fn field_write(&self, field: &FieldUnit, value: WrappedObject, depth: usize) -> Result<(), AmlError> {
        trace!("AML field write. Field = {:?}", field);

        let integer_bytes;
        let value_bytes = match &*value {
            Object::Integer(value) => {
                integer_bytes = value.to_le_bytes();
                &integer_bytes as &[u8]
            }
            Object::Buffer(bytes) => bytes,
            Object::String(string) => string.as_bytes(),
            _ => Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Integer, got: value.typ() })?,
        };
        let access_width_bits = field.flags.access_type_bytes()? * 8;

        let mut written_so_far = 0;
        let mut i = 0;
        while written_so_far < field.bit_length {
            let bit =
                field.bit_index.checked_add(i * access_width_bits).ok_or(AmlError::RegionAccessOutOfBounds)?;
            let aligned_offset = object::align_down(bit, access_width_bits);
            let dst_index = if i == 0 { field.bit_index % access_width_bits } else { 0 };
            let length = usize::min(field.bit_length - written_so_far, access_width_bits - dst_index);

            /*
             * If we're not going to write a whole native access, respect the field's
             * update rule. If we're meant to preserve the surrounding bits, we need to do
             * a read first.
             */
            let mut bytes = if length < access_width_bits {
                match field.flags.update_rule() {
                    FieldUpdateRule::Preserve => self
                        .field_access(field, aligned_offset / 8, access_width_bits / 8, None, depth)?
                        .to_le_bytes(),
                    FieldUpdateRule::WriteAsOnes => [0xff; 8],
                    FieldUpdateRule::WriteAsZeros => [0; 8],
                }
            } else {
                [0; 8]
            };

            object::copy_bits(value_bytes, written_so_far, &mut bytes, dst_index, length);
            self.field_access(
                field,
                aligned_offset / 8,
                access_width_bits / 8,
                Some(u64::from_le_bytes(bytes)),
                depth,
            )?;
            written_so_far += length;
            i += 1;
        }

        Ok(())
    }

    /// Hardening: one access of a field's backing store: the region of a normal field; the region after the bank field
    /// was set for a bank field; the data field after the index field was set to the byte offset for an index
    /// field (both were `todo!()`). Reads when `write` is `None`. Index and bank fields nest at most a few deep.
    fn field_access(
        &self,
        field: &FieldUnit,
        offset: usize,
        length: usize,
        write: Option<u64>,
        depth: usize,
    ) -> Result<u64, AmlError> {
        if depth > 4 {
            return Err(AmlError::LimitExceeded(Limit::Reentrancy));
        }
        self.ops_used.fetch_add(1, Ordering::AcqRel);
        if self.ops_used.load(Ordering::Acquire) > self.limits().max_ops {
            return Err(AmlError::LimitExceeded(Limit::Ops));
        }
        let region = match field.kind {
            FieldUnitKind::Normal { ref region } => region,
            FieldUnitKind::Bank { ref region, ref bank, bank_value } => {
                let Object::FieldUnit(ref bank) = **bank else {
                    return Err(AmlError::ObjectNotOfExpectedType {
                        expected: ObjectType::FieldUnit,
                        got: bank.typ(),
                    });
                };
                self.field_write(bank, Object::Integer(bank_value).wrap(), depth + 1)?;
                region
            }
            FieldUnitKind::Index { ref index, ref data } => {
                let (Object::FieldUnit(index), Object::FieldUnit(data)) = (&**index, &**data) else {
                    return Err(AmlError::ObjectNotOfExpectedType {
                        expected: ObjectType::FieldUnit,
                        got: index.typ(),
                    });
                };
                self.field_write(index, Object::Integer(offset as u64).wrap(), depth + 1)?;
                return match write {
                    Some(value) => self.field_write(data, Object::Integer(value).wrap(), depth + 1).map(|()| 0),
                    None => self.field_read(data, depth + 1)?.as_integer(),
                };
            }
        };
        // Hardening: was `panic!()`.
        let Object::OpRegion(ref region) = **region else {
            return Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::OpRegion, got: region.typ() });
        };
        match write {
            Some(value) => self.do_native_region_write(region, offset, length, value).map(|()| 0),
            None => self.do_native_region_read(region, offset, length),
        }
    }

    /// Hardening: checks an access against the region's length.
    fn check_region_access(region: &OpRegion, offset: usize, length: usize) -> Result<(), AmlError> {
        let end = (offset as u64).checked_add(length as u64).ok_or(AmlError::RegionAccessOutOfBounds)?;
        if end > region.length { Err(AmlError::RegionAccessOutOfBounds) } else { Ok(()) }
    }

    /// Performs an actual read from an operation region. `offset` and `length` must respect the
    /// access requirements of the field being read, and are supplied in **bytes**. This may call
    /// AML methods if required, and may invoke user-supplied handlers.
    ///
    /// Hardening: an installed handler for the region's space is used first (for every space); unsupported widths and
    /// accesses past the region are errors, not panics.
    fn do_native_region_read(&self, region: &OpRegion, offset: usize, length: usize) -> Result<u64, AmlError> {
        trace!("Native field read. Region = {:?}, offset = {:#x}, length={:#x}", region, offset, length);
        Self::check_region_access(region, offset, length)?;

        if let Some(handler) = self.region_handler(region.space) {
            return handler.read(region, offset as u64, (length * 8) as u8);
        }

        match region.space {
            RegionSpace::SystemMemory => {
                let address = self.memory_address(region, offset)?;
                Ok(match length {
                    1 => self.handler.read_u8(address) as u64,
                    2 => self.handler.read_u16(address) as u64,
                    4 => self.handler.read_u32(address) as u64,
                    8 => self.handler.read_u64(address),
                    _ => return Err(AmlError::InvalidFieldFlags),
                })
            }
            RegionSpace::SystemIO => {
                let port = Self::io_port(region, offset)?;
                Ok(match length {
                    1 => self.handler.read_io_u8(port) as u64,
                    2 => self.handler.read_io_u16(port) as u64,
                    4 => self.handler.read_io_u32(port) as u64,
                    _ => return Err(AmlError::InvalidFieldFlags),
                })
            }
            RegionSpace::PciConfig => {
                let address = self.pci_address_for_device(&region.parent_device_path)?;
                let offset = Self::pci_offset(region, offset, length)?;
                match length {
                    1 => Ok(self.handler.read_pci_u8(address, offset) as u64),
                    2 => Ok(self.handler.read_pci_u16(address, offset) as u64),
                    4 => Ok(self.handler.read_pci_u32(address, offset) as u64),
                    _ => Err(AmlError::InvalidFieldFlags),
                }
            }
            other => Err(AmlError::NoHandlerForRegionAccess(other)),
        }
    }

    /// Performs an actual write to an operation region. `offset` and `length` must respect the
    /// access requirements of the field being read, and are supplied in **bytes**. This may call
    /// AML methods if required, and may invoke user-supplied handlers.
    fn do_native_region_write(
        &self,
        region: &OpRegion,
        offset: usize,
        length: usize,
        value: u64,
    ) -> Result<(), AmlError> {
        trace!(
            "Native field write. Region = {:?}, offset = {:#x}, length={:#x}, value={:#x}",
            region, offset, length, value
        );
        Self::check_region_access(region, offset, length)?;

        if let Some(handler) = self.region_handler(region.space) {
            return handler.write(region, offset as u64, (length * 8) as u8, value);
        }

        match region.space {
            RegionSpace::SystemMemory => {
                let address = self.memory_address(region, offset)?;
                match length {
                    1 => self.handler.write_u8(address, value as u8),
                    2 => self.handler.write_u16(address, value as u16),
                    4 => self.handler.write_u32(address, value as u32),
                    8 => self.handler.write_u64(address, value),
                    _ => return Err(AmlError::InvalidFieldFlags),
                }
                Ok(())
            }
            RegionSpace::SystemIO => {
                let port = Self::io_port(region, offset)?;
                match length {
                    1 => self.handler.write_io_u8(port, value as u8),
                    2 => self.handler.write_io_u16(port, value as u16),
                    4 => self.handler.write_io_u32(port, value as u32),
                    _ => return Err(AmlError::InvalidFieldFlags),
                }
                Ok(())
            }
            RegionSpace::PciConfig => {
                let address = self.pci_address_for_device(&region.parent_device_path)?;
                let offset = Self::pci_offset(region, offset, length)?;
                match length {
                    1 => self.handler.write_pci_u8(address, offset, value as u8),
                    2 => self.handler.write_pci_u16(address, offset, value as u16),
                    4 => self.handler.write_pci_u32(address, offset, value as u32),
                    _ => return Err(AmlError::InvalidFieldFlags),
                }
                Ok(())
            }
            other => Err(AmlError::NoHandlerForRegionAccess(other)),
        }
    }

    /// Hardening: `base + offset` with no overflow.
    fn memory_address(&self, region: &OpRegion, offset: usize) -> Result<usize, AmlError> {
        region
            .base
            .checked_add(offset as u64)
            .and_then(|address| usize::try_from(address).ok())
            .ok_or(AmlError::RegionAccessOutOfBounds)
    }

    /// Hardening: a port above 0xffff is an error (it wrapped).
    fn io_port(region: &OpRegion, offset: usize) -> Result<u16, AmlError> {
        region
            .base
            .checked_add(offset as u64)
            .and_then(|port| u16::try_from(port).ok())
            .ok_or(AmlError::RegionAccessOutOfBounds)
    }

    /// Hardening: the configuration space is 4 KiB.
    fn pci_offset(region: &OpRegion, offset: usize, length: usize) -> Result<u16, AmlError> {
        let start = region.base.checked_add(offset as u64).ok_or(AmlError::RegionAccessOutOfBounds)?;
        match start.checked_add(length as u64) {
            Some(end) if end <= 4096 => Ok(start as u16),
            _ => Err(AmlError::RegionAccessOutOfBounds),
        }
    }

    fn pci_address_for_device(&self, path: &AmlName) -> Result<PciAddress, AmlError> {
        /*
         * TODO: it's not ideal to do these reads for every native access. See if we can
         * cache them somewhere?
         */
        let seg = match self.evaluate_if_present(static_name("_SEG").resolve(path)?, vec![])? {
            Some(value) => value.as_integer()?,
            None => 0,
        };
        let bus = match self.evaluate_if_present(static_name("_BBR").resolve(path)?, vec![])? {
            Some(value) => value.as_integer()?,
            None => 0,
        };
        let (device, function) = {
            let adr = self.evaluate_if_present(static_name("_ADR").resolve(path)?, vec![])?;
            let adr = match adr {
                Some(adr) => adr.as_integer()?,
                None => 0,
            };
            (adr.get_bits(16..32), adr.get_bits(0..16))
        };
        Ok(PciAddress::new(seg as u16, bus as u8, device as u8, function as u8))
    }
}

/// Hardening: a package's element count, against the limit.
fn check_elements(limits: &Limits, count: u64) -> Result<usize, AmlError> {
    match usize::try_from(count) {
        Ok(count) if count <= limits.max_package_elements => Ok(count),
        _ => Err(AmlError::LimitExceeded(Limit::PackageElements)),
    }
}

/// A `MethodContext` represents a piece of running AML code - either a real method, or the
/// top-level of an AML table.
///
/// ### Safety
/// `MethodContext` does not keep the lifetime of the underlying AML stream, which for tables is
/// borrowed from the underlying physical mapping. This is because the interpreter needs to
/// preempt method contexts that execute other methods, and these contexts may have disparate
/// lifetimes. This is made safe in the case of methods by the context holding a reference to the
/// method object, but must be handled manually for AML tables.
struct MethodContext {
    current_block: Block,
    block_stack: Vec<Block>,
    in_flight: Vec<OpInFlight>,
    args: [WrappedObject; 8],
    locals: [WrappedObject; 8],
    current_scope: AmlName,

    _method: Option<WrappedObject>,
}

struct Block {
    stream: *const [u8],
    pc: usize,
    kind: BlockKind,
}

impl Block {
    fn stream(&self) -> &[u8] {
        unsafe { &*self.stream }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum BlockKind {
    Table,
    Method {
        method_scope: AmlName,
    },
    Scope {
        old_scope: AmlName,
    },
    Package,
    VarPackage,
    /// Used for executing the then-branch of an `DefIfElse`. After finishing, it will check for
    /// and skip over an else-branch, if present.
    IfThenBranch,
    While {
        start_pc: usize,
        // Hardening: how many times the loop went round (bounded by `Limits::max_loop_iterations`).
        iterations: u64,
    },
}

/// A `ResolveBehaviour` describes how a name at the top-level should be resolved as part of an
/// operation.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ResolveBehaviour {
    /// Attempt to resolve the name to an object that has already been defined. There are generally
    /// no forward definitions in AML, so this is the usual resolution behaviour for most operands.
    TermArg,
    /// Resolve a name to reference an object. This is used when an operation needs to operate on
    /// the object itself, rather than evaluate it to a value. For example, accessing a `FieldUnit` would
    /// read a value from the field with `TermArg`, but resolves to the `FieldUnit` with
    /// this behaviour.
    SuperName,
    /// Behaves the same as `SuperName` if the object exists, but resolves successfully to an
    /// unresolved reference if the object does not exist. Used by `DefCondRefOf`.
    SuperNameIfExists,
    /// `SuperName`, but can also be `NullName`
    Target,
    /// Surrogate argument, used by `DefPackage` and `DefVarPackage`. Only one of these is emitted,
    /// but represents parsing of potentially many package elements. Names in packages should be
    /// resolved into `String` objects - this is not well defined by the specification, but matches
    /// expected behaviour of other interpreters.
    AsPackageElements,
    /// Used with [`OpInFlight::new_with`] to represent arguments that have already been resolved
    /// when an operation enters flight.
    Placeholder,
}

#[derive(Debug)]
struct OpInFlight {
    op: Opcode,
    expected_arguments: usize,
    arguments: Vec<Argument>,
    resolve_behaviour: &'static [ResolveBehaviour],
    // Hardening: the resolved path of the object a `Notify` names.
    target_name: Option<AmlName>,
}

#[derive(Debug)]
enum Argument {
    Null,
    Object(WrappedObject),
    Namestring(AmlName),
    ByteData(u8),
    DWordData(u32),
    TrackedPc(usize),
    PkgLength(usize),
}

impl OpInFlight {
    /// Creates a new `OpInFlight`. The number of expected arguments is inferred from the number of
    /// `ResolveBehaviour`s passed.
    pub fn new(op: Opcode, resolve_behaviour: &'static [ResolveBehaviour]) -> OpInFlight {
        OpInFlight {
            op,
            expected_arguments: resolve_behaviour.len(),
            arguments: Vec::new(),
            resolve_behaviour,
            target_name: None,
        }
    }

    /// Creates a new `OpInFlight` with the given number of expected arguments. This should be used
    /// when the correct number of expected arguments differs from the number of
    /// `ResolveBehaviour`s passed.
    pub fn new_dynamic(
        op: Opcode,
        expected_arguments: usize,
        resolve_behaviour: &'static [ResolveBehaviour],
    ) -> OpInFlight {
        OpInFlight { op, expected_arguments, arguments: Vec::new(), resolve_behaviour, target_name: None }
    }

    /// Creates a new `OpInFlight` with a number of arguments that have already been interpreted,
    /// and is expecting some `more` arguments.
    pub fn new_with_dynamic(
        op: Opcode,
        arguments: Vec<Argument>,
        more: usize,
        resolve_behaviour: &'static [ResolveBehaviour],
    ) -> OpInFlight {
        OpInFlight {
            op,
            expected_arguments: arguments.len() + more,
            arguments,
            resolve_behaviour,
            target_name: None,
        }
    }

    /// Creates a new `OpInFlight` with a number of arguments that have already been interpreted,
    /// and is expecting more - the number of remaining arguments is inferred from the number of
    /// `ResolveBehaviour`s passed (with existing arguments marked as
    /// `ResolveBehaviour::Placeholder`).
    pub fn new_with(
        op: Opcode,
        arguments: Vec<Argument>,
        resolve_behaviour: &'static [ResolveBehaviour],
    ) -> OpInFlight {
        OpInFlight {
            op,
            expected_arguments: resolve_behaviour.len(),
            arguments,
            resolve_behaviour,
            target_name: None,
        }
    }

    /// Get the desired `ResolveBehaviour` for the argument currently being interpreted
    /// Hardening: an op given more operands than it takes is malformed AML (this was a `panic!`).
    fn resolve_behaviour(&self) -> Result<ResolveBehaviour, AmlError> {
        if let Some(behaviour) = self.resolve_behaviour.get(self.arguments.len()) {
            Ok(*behaviour)
        } else if self.op == Opcode::Package || (self.op == Opcode::VarPackage && !self.arguments.is_empty()) {
            Ok(ResolveBehaviour::AsPackageElements)
        } else {
            Err(AmlError::MalformedStream)
        }
    }
}

impl MethodContext {
    unsafe fn new_from_table(stream: &[u8]) -> MethodContext {
        let block = Block { stream: stream as *const [u8], pc: 0, kind: BlockKind::Table };
        MethodContext {
            current_block: block,
            block_stack: Vec::new(),
            in_flight: Vec::new(),
            args: core::array::from_fn(|_| Object::Uninitialized.wrap()),
            locals: core::array::from_fn(|_| Object::Uninitialized.wrap()),
            current_scope: AmlName::root(),
            _method: None,
        }
    }

    fn new_from_method(
        method: WrappedObject,
        args: Vec<WrappedObject>,
        scope: AmlName,
    ) -> Result<MethodContext, AmlError> {
        if let Object::Method { code, flags } = &*method {
            if args.len() != flags.arg_count() {
                return Err(AmlError::MethodArgCountIncorrect);
            }
            let block = Block {
                stream: code as &[u8] as *const [u8],
                pc: 0,
                kind: BlockKind::Method { method_scope: scope.clone() },
            };
            let args = core::array::from_fn(|i| {
                if let Some(arg) = args.get(i) { arg.clone() } else { Object::Uninitialized.wrap() }
            });
            let context = MethodContext {
                current_block: block,
                block_stack: Vec::new(),
                in_flight: Vec::new(),
                args,
                locals: core::array::from_fn(|_| Object::Uninitialized.wrap()),
                current_scope: scope,
                _method: Some(method.clone()),
            };
            Ok(context)
        } else {
            Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Method, got: method.typ() })
        }
    }

    fn last_op(&mut self) -> Result<&mut OpInFlight, AmlError> {
        match self.in_flight.last_mut() {
            Some(op) => Ok(op),
            None => Err(AmlError::NoCurrentOp),
        }
    }

    fn contribute_arg(&mut self, arg: Argument) {
        if let Some(in_flight) = self.in_flight.last_mut()
            && in_flight.arguments.len() < in_flight.expected_arguments
        {
            in_flight.arguments.push(arg);
        }
    }

    /// Start a new `InFlightOp`.
    fn start(&mut self, op: OpInFlight) {
        trace!(
            "START OP: {:?}, args: {:?}, with {} more needed ({:?})",
            op.op,
            op.arguments,
            op.expected_arguments.saturating_sub(op.arguments.len()),
            op.resolve_behaviour
        );
        self.in_flight.push(op);
    }

    fn retire_op(&mut self, op: OpInFlight) {
        trace!("RETIRE OP: {:?}, args: {:?}", op.op, op.arguments);
    }

    /// Hardening: a block past the end of the stream, or nested deeper than the limit, is an error (the slice panicked
    /// and the nesting was unbounded).
    fn start_new_block(&mut self, kind: BlockKind, length: usize, limits: &Limits) -> Result<(), AmlError> {
        if self.block_stack.len() >= limits.max_block_nesting {
            return Err(AmlError::LimitExceeded(Limit::BlockNesting));
        }
        let end = self.current_block.pc.checked_add(length).ok_or(AmlError::MalformedStream)?;
        let stream = self.current_block.stream().get(..end).ok_or(AmlError::MalformedStream)? as *const [u8];
        let block = Block { stream, pc: self.current_block.pc, kind };
        self.current_block.pc = end;
        self.block_stack.push(mem::replace(&mut self.current_block, block));
        Ok(())
    }

    /// Hardening: leaves the current block for the one around it (each caller used `pop().unwrap()`).
    fn pop_block(&mut self) -> Result<(), AmlError> {
        self.current_block = self.block_stack.pop().ok_or(AmlError::MalformedStream)?;
        Ok(())
    }

    /// Hardening: what is left of a PkgLength that started at `start_pc` (the subtraction underflowed).
    fn remaining(&self, start_pc: usize, pkg_length: usize) -> Result<usize, AmlError> {
        let used = self.current_block.pc.checked_sub(start_pc).ok_or(AmlError::MalformedStream)?;
        pkg_length.checked_sub(used).ok_or(AmlError::MalformedStream)
    }

    /// Hardening: the next `length` bytes of the stream, bounds-checked.
    fn take(&mut self, length: usize) -> Result<&[u8], AmlError> {
        let start = self.current_block.pc;
        let end = start.checked_add(length).ok_or(AmlError::MalformedStream)?;
        if end > self.current_block.stream().len() {
            return Err(AmlError::MalformedStream);
        }
        self.current_block.pc = end;
        Ok(&self.current_block.stream()[start..end])
    }

    /// Hardening: skips `length` bytes, bounds-checked.
    fn skip(&mut self, length: usize) -> Result<(), AmlError> {
        self.take(length).map(|_| ())
    }

    /// Hardening: the scope of the method this context runs, if it runs one.
    fn method_scope(&self) -> Option<AmlName> {
        self.block_stack.iter().chain(core::iter::once(&self.current_block)).find_map(|block| match block.kind {
            BlockKind::Method { ref method_scope } => Some(method_scope.clone()),
            _ => None,
        })
    }

    fn opcode(&mut self) -> Result<Opcode, AmlError> {
        let opcode: u16 = match self.next()? {
            0x5b => {
                // Hardening: an extended-opcode prefix with nothing after it is malformed, not the end of the block.
                let ext = self.next().map_err(|_| AmlError::MalformedStream)?;
                (0x5b << 8) as u16 | ext as u16
            }
            other => other as u16,
        };

        Ok(match opcode {
            0x00 => Opcode::Zero,
            0x01 => Opcode::One,
            0x06 => Opcode::Alias,
            0x08 => Opcode::Name,
            0x0a => Opcode::BytePrefix,
            0x0b => Opcode::WordPrefix,
            0x0c => Opcode::DWordPrefix,
            0x0d => Opcode::StringPrefix,
            0x0e => Opcode::QWordPrefix,
            0x10 => Opcode::Scope,
            0x11 => Opcode::Buffer,
            0x12 => Opcode::Package,
            0x13 => Opcode::VarPackage,
            0x14 => Opcode::Method,
            0x15 => Opcode::External,
            0x2e => Opcode::DualNamePrefix,
            0x2f => Opcode::MultiNamePrefix,
            0x30..=0x39 => Opcode::Digit(opcode as u8),    // b'0'..=b'9'
            0x41..=0x5a => Opcode::NameChar(opcode as u8), // b'A'..=b'Z'
            0x5b01 => Opcode::Mutex,
            0x5b02 => Opcode::Event,
            0x5b12 => Opcode::CondRefOf,
            0x5b13 => Opcode::CreateField,
            0x5b1f => Opcode::LoadTable,
            0x5b20 => Opcode::Load,
            0x5b21 => Opcode::Stall,
            0x5b22 => Opcode::Sleep,
            0x5b23 => Opcode::Acquire,
            0x5b24 => Opcode::Signal,
            0x5b25 => Opcode::Wait,
            0x5b26 => Opcode::Reset,
            0x5b27 => Opcode::Release,
            0x5b28 => Opcode::FromBCD,
            0x5b29 => Opcode::ToBCD,
            0x5b30 => Opcode::Revision,
            0x5b31 => Opcode::Debug,
            0x5b32 => Opcode::Fatal,
            0x5b33 => Opcode::Timer,
            0x5b80 => Opcode::OpRegion,
            0x5b81 => Opcode::Field,
            0x5b82 => Opcode::Device,
            0x5b83 => Opcode::Processor,
            0x5b84 => Opcode::PowerRes,
            0x5b85 => Opcode::ThermalZone,
            0x5b86 => Opcode::IndexField,
            0x5b87 => Opcode::BankField,
            0x5b88 => Opcode::DataRegion,
            0x5c => Opcode::RootChar,
            0x5e => Opcode::ParentPrefixChar,
            0x5f => Opcode::NameChar(b'_'),
            0x60..=0x67 => Opcode::Local(opcode as u8 - 0x60),
            0x68..=0x6e => Opcode::Arg(opcode as u8 - 0x68),
            0x70 => Opcode::Store,
            0x71 => Opcode::RefOf,
            0x72 => Opcode::Add,
            0x73 => Opcode::Concat,
            0x74 => Opcode::Subtract,
            0x75 => Opcode::Increment,
            0x76 => Opcode::Decrement,
            0x77 => Opcode::Multiply,
            0x78 => Opcode::Divide,
            0x79 => Opcode::ShiftLeft,
            0x7a => Opcode::ShiftRight,
            0x7b => Opcode::And,
            0x7c => Opcode::Nand,
            0x7d => Opcode::Or,
            0x7e => Opcode::Nor,
            0x7f => Opcode::Xor,
            0x80 => Opcode::Not,
            0x81 => Opcode::FindSetLeftBit,
            0x82 => Opcode::FindSetRightBit,
            0x83 => Opcode::DerefOf,
            0x84 => Opcode::ConcatRes,
            0x85 => Opcode::Mod,
            0x86 => Opcode::Notify,
            0x87 => Opcode::SizeOf,
            0x88 => Opcode::Index,
            0x89 => Opcode::Match,
            0x8a => Opcode::CreateDWordField,
            0x8b => Opcode::CreateWordField,
            0x8c => Opcode::CreateByteField,
            0x8d => Opcode::CreateBitField,
            0x8e => Opcode::ObjectType,
            0x8f => Opcode::CreateQWordField,
            0x90 => Opcode::LAnd,
            0x91 => Opcode::LOr,
            /*
             * `0x92` is a bit strange. It can be an opcode in its own right (`LNotOp`), but when
             * followed by `0x93..=0x95`, it instead serves as a negating prefix to encode
             * `LNotEqualOp`, `LLessEqualOp`, and `LGreaterEqualOp`.
             */
            0x92 => match self.peek() {
                Ok(0x93) => {
                    self.current_block.pc += 1;
                    Opcode::LNotEqual
                }
                Ok(0x94) => {
                    self.current_block.pc += 1;
                    Opcode::LLessEqual
                }
                Ok(0x95) => {
                    self.current_block.pc += 1;
                    Opcode::LGreaterEqual
                }
                _ => Opcode::LNot,
            },
            0x93 => Opcode::LEqual,
            0x94 => Opcode::LGreater,
            0x95 => Opcode::LLess,
            0x96 => Opcode::ToBuffer,
            0x97 => Opcode::ToDecimalString,
            0x98 => Opcode::ToHexString,
            0x99 => Opcode::ToInteger,
            0x9c => Opcode::ToString,
            0x9d => Opcode::CopyObject,
            0x9e => Opcode::Mid,
            0x9f => Opcode::Continue,
            0xa0 => Opcode::If,
            0xa1 => Opcode::Else,
            0xa2 => Opcode::While,
            0xa3 => Opcode::Noop,
            0xa4 => Opcode::Return,
            0xa5 => Opcode::Break,
            0xcc => Opcode::Breakpoint,
            0xff => Opcode::Ones,

            _ => Err(AmlError::IllegalOpcode(opcode))?,
        })
    }

    fn pkglength(&mut self) -> Result<usize, AmlError> {
        let lead_byte = self.next()?;
        // (Two bits: always below 4. The `assert!` that said so is gone.)
        let byte_count = lead_byte.get_bits(6..8);

        if byte_count == 0 {
            Ok(lead_byte.get_bits(0..6) as usize)
        } else {
            let mut length = lead_byte.get_bits(0..4) as usize;
            for i in 0..byte_count {
                length |= (self.next()? as usize) << (4 + i * 8);
            }
            Ok(length)
        }
    }

    fn namestring(&mut self) -> Result<AmlName, AmlError> {
        use namespace::{NameComponent, NameSeg};

        /*
         * The NameString grammar is actually a little finicky and annoying.
         *
         * NameString := <RootChar NamePath> | <PrefixPath NamePath>
         * PrefixPath := Nothing | <'^' PrefixPath>
         * NamePath := NameSeg | DualNamePath | MultiNamePath | NullName
         * DualNamePath := DualNamePrefix NameSeg NameSeg
         * MultiNamePath := MultiNamePrefix SegCount NameSeg(SegCount)
         */
        const NULL_NAME: u8 = 0x00;
        const DUAL_NAME_PREFIX: u8 = 0x2e;
        const MULTI_NAME_PREFIX: u8 = 0x2f;

        let mut components = vec![];

        match self.peek()? {
            b'\\' => {
                self.next()?;
                components.push(NameComponent::Root);
            }
            b'^' => {
                components.push(NameComponent::Prefix);
                self.next()?;
                while self.peek()? == b'^' {
                    self.next()?;
                    components.push(NameComponent::Prefix);
                }
            }
            _ => (),
        }

        let next = self.next()?;
        match next {
            NULL_NAME => {}
            DUAL_NAME_PREFIX => {
                for _ in 0..2 {
                    let name_seg = [self.next()?, self.next()?, self.next()?, self.next()?];
                    components.push(NameComponent::Segment(NameSeg::from_bytes(name_seg)?));
                }
            }
            MULTI_NAME_PREFIX => {
                let count = self.next()?;
                for _ in 0..count {
                    let name_seg = [self.next()?, self.next()?, self.next()?, self.next()?];
                    components.push(NameComponent::Segment(NameSeg::from_bytes(name_seg)?));
                }
            }
            first_char => {
                if !namespace::is_lead_name_char(first_char) {
                    self.current_block.pc -= 1;
                }
                let name_seg = [first_char, self.next()?, self.next()?, self.next()?];
                components.push(namespace::NameComponent::Segment(namespace::NameSeg::from_bytes(name_seg)?));
            }
        }

        Ok(AmlName::from_components(components))
    }

    fn next(&mut self) -> Result<u8, AmlError> {
        if self.current_block.pc >= self.current_block.stream.len() {
            return Err(AmlError::RunOutOfStream);
        }

        let byte = self.current_block.stream()[self.current_block.pc];
        self.current_block.pc += 1;

        Ok(byte)
    }

    fn next_u16(&mut self) -> Result<u16, AmlError> {
        Ok(u16::from_le_bytes([self.next()?, self.next()?]))
    }

    fn next_u32(&mut self) -> Result<u32, AmlError> {
        Ok(u32::from_le_bytes([self.next()?, self.next()?, self.next()?, self.next()?]))
    }

    fn next_u64(&mut self) -> Result<u64, AmlError> {
        Ok(u64::from_le_bytes([
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
        ]))
    }

    fn peek(&self) -> Result<u8, AmlError> {
        if self.current_block.pc >= self.current_block.stream.len() {
            return Err(AmlError::RunOutOfStream);
        }

        Ok(self.current_block.stream()[self.current_block.pc])
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Opcode {
    Zero,
    One,
    Alias,
    Name,
    BytePrefix,
    WordPrefix,
    DWordPrefix,
    StringPrefix,
    QWordPrefix,
    Scope,
    Buffer,
    Package,
    VarPackage,
    Method,
    External,
    DualNamePrefix,
    MultiNamePrefix,
    Digit(u8),
    NameChar(u8),
    Mutex,
    Event,
    CondRefOf,
    CreateField,
    LoadTable,
    Load,
    Stall,
    Sleep,
    Acquire,
    Signal,
    Wait,
    Reset,
    Release,
    FromBCD,
    ToBCD,
    Revision,
    Debug,
    Fatal,
    Timer,
    OpRegion,
    Field,
    Device,
    Processor,
    PowerRes,
    ThermalZone,
    IndexField,
    BankField,
    DataRegion,
    RootChar,
    ParentPrefixChar,
    Local(u8),
    Arg(u8),
    Store,
    RefOf,
    Add,
    Concat,
    Subtract,
    Increment,
    Decrement,
    Multiply,
    Divide,
    ShiftLeft,
    ShiftRight,
    And,
    Nand,
    Or,
    Nor,
    Xor,
    Not,
    FindSetLeftBit,
    FindSetRightBit,
    DerefOf,
    ConcatRes,
    Mod,
    Notify,
    SizeOf,
    Index,
    Match,
    CreateDWordField,
    CreateWordField,
    CreateByteField,
    CreateBitField,
    ObjectType,
    CreateQWordField,
    LAnd,
    LOr,
    LNot,
    LNotEqual,
    LLessEqual,
    LGreaterEqual,
    LEqual,
    LGreater,
    LLess,
    ToBuffer,
    ToDecimalString,
    ToHexString,
    ToInteger,
    ToString,
    CopyObject,
    Mid,
    Continue,
    If,
    Else,
    While,
    Noop,
    Return,
    Break,
    Breakpoint,
    Ones,

    /*
     * Internal opcodes are not produced from the bytecode, but are used to track special in-flight
     * ops etc.
     */
    InternalMethodCall,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Operation {
    Mid,
    SizeOf,
    Acquire,
    Release,
    ConvertToBuffer,

    ToBuffer,
    ToInteger,
    ToString,
    ToDecOrHexString,

    ReadBufferField,
    WriteBufferField,
    LogicalOp,
    DecodePrt,
    ParseResource,

    ResetEvent,
    SignalEvent,
    WaitEvent,
    Store,
}

#[derive(Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum AmlError {
    RunOutOfStream,
    IllegalOpcode(u16),
    InvalidFieldFlags,

    InvalidName(Option<AmlName>),

    InvalidNameSeg([u8; 4]),
    InvalidNormalizedName(AmlName),
    RootHasNoParent,
    EmptyNamesAreInvalid,
    LevelDoesNotExist(AmlName),
    NameCollision(AmlName),
    ObjectDoesNotExist(AmlName),

    NoCurrentOp,
    ElseFoundWithoutCorrespondingIf,
    ContinueOutsideOfWhile,
    BreakOutsideOfWhile,

    MethodArgCountIncorrect,

    InvalidOperationOnObject {
        op: Operation,
        typ: ObjectType,
    },
    IndexOutOfBounds,
    ObjectNotOfExpectedType {
        expected: ObjectType,
        got: ObjectType,
    },

    InvalidResourceDescriptor,
    UnexpectedResourceType,

    NoHandlerForRegionAccess(RegionSpace),
    MutexAcquireTimeout,

    PrtInvalidAddress,
    PrtInvalidPin,
    PrtInvalidGsi,
    PrtInvalidSource,
    PrtNoEntry,

    /// This is emitted to signal that the library does not support the requested behaviour. This
    /// should eventually never be emitted.
    LibUnimplemented,

    // Hardening: the errors that replaced panics.
    /// An opcode or feature this interpreter does not run (`Load`, `LoadTable`, `Match`).
    Unsupported(&'static str),
    /// The stream is malformed: a length past its end, an op given the wrong operands, a block that does not
    /// close.
    MalformedStream,
    /// A [`Limits`] limit was reached.
    LimitExceeded(Limit),
    /// `Divide` or `Mod` by zero.
    DivideByZero,
    /// A handler for this region space is installed already.
    RegionHandlerAlreadyInstalled(RegionSpace),
    /// A field access past its operation region (or past the address space).
    RegionAccessOutOfBounds,
    /// A fixed ACPI register could not be accessed.
    RegisterAccessFailed,
    /// The AML ran `Fatal`: the host was told (`Handler::handle_fatal_error`) and the evaluation stops.
    FatalOpcode {
        typ: u8,
        code: u32,
        arg: u64,
    },
}

/// Hardening: a buffer field of `length` bits at bit `offset` must lie inside its buffer (or string) as it is now, and have
/// at least one bit. (A buffer replaced by a shorter one later is checked again when the field is read or written.)
fn field_in_buffer(buffer: &WrappedObject, offset: usize, length: usize) -> Result<(), AmlError> {
    let bytes = match **buffer {
        Object::Buffer(ref b) => b.len(),
        Object::String(ref s) => s.len(),
        ref other => {
            return Err(AmlError::InvalidOperationOnObject { op: Operation::ReadBufferField, typ: other.typ() });
        }
    };
    match offset.checked_add(length) {
        Some(end) if length > 0 && end <= bytes.saturating_mul(8) => Ok(()),
        _ => Err(AmlError::IndexOutOfBounds),
    }
}
