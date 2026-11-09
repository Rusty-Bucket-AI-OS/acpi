// Hardening: helpers for the hardening tests: a host `Handler` that records what the AML asked of it, a fake embedded
// controller, and builders for hand-assembled AML (no iasl needed).

#![allow(dead_code)]

use acpi::{
    Handle,
    Handler,
    PciAddress,
    PhysicalMapping,
    aml::{
        AmlError,
        Interpreter,
        Limits,
        namespace::AmlName,
        op_region::{OpRegion, RegionHandler},
    },
};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct Record {
    pub notifies: Vec<(String, u64)>,
    pub fatals: Vec<(u8, u32, u64)>,
    pub sleeps: Vec<u64>,
    pub mutexes: u32,
    /// The host's clock, in nanoseconds: a `Sleep` moves it on, nothing else does.
    pub clock_ns: u64,
}

/// A host with no hardware: memory and ports read as zero, writes go nowhere, and notifications are recorded.
#[derive(Clone, Default)]
pub struct TestHandler(pub Arc<Mutex<Record>>);

impl Handler for TestHandler {
    unsafe fn map_physical_region<T>(&self, _physical_address: usize, _size: usize) -> PhysicalMapping<Self, T> {
        unimplemented!("the bare interpreter maps nothing")
    }
    fn unmap_physical_region<T>(_region: &PhysicalMapping<Self, T>) {}
    fn read_u8(&self, _address: usize) -> u8 {
        0
    }
    fn read_u16(&self, _address: usize) -> u16 {
        0
    }
    fn read_u32(&self, _address: usize) -> u32 {
        0
    }
    fn read_u64(&self, _address: usize) -> u64 {
        0
    }
    fn write_u8(&self, _address: usize, _value: u8) {}
    fn write_u16(&self, _address: usize, _value: u16) {}
    fn write_u32(&self, _address: usize, _value: u32) {}
    fn write_u64(&self, _address: usize, _value: u64) {}
    fn read_io_u8(&self, _port: u16) -> u8 {
        0
    }
    fn read_io_u16(&self, _port: u16) -> u16 {
        0
    }
    fn read_io_u32(&self, _port: u16) -> u32 {
        0
    }
    fn write_io_u8(&self, _port: u16, _value: u8) {}
    fn write_io_u16(&self, _port: u16, _value: u16) {}
    fn write_io_u32(&self, _port: u16, _value: u32) {}
    fn read_pci_u8(&self, _address: PciAddress, _offset: u16) -> u8 {
        0
    }
    fn read_pci_u16(&self, _address: PciAddress, _offset: u16) -> u16 {
        0
    }
    fn read_pci_u32(&self, _address: PciAddress, _offset: u16) -> u32 {
        0
    }
    fn write_pci_u8(&self, _address: PciAddress, _offset: u16, _value: u8) {}
    fn write_pci_u16(&self, _address: PciAddress, _offset: u16, _value: u16) {}
    fn write_pci_u32(&self, _address: PciAddress, _offset: u16, _value: u32) {}
    fn nanos_since_boot(&self) -> u64 {
        self.0.lock().unwrap().clock_ns
    }
    fn stall(&self, _microseconds: u64) {}
    fn sleep(&self, milliseconds: u64) {
        let mut record = self.0.lock().unwrap();
        record.sleeps.push(milliseconds);
        record.clock_ns = record.clock_ns.saturating_add(milliseconds.saturating_mul(1_000_000));
    }
    fn create_mutex(&self) -> Handle {
        let mut record = self.0.lock().unwrap();
        record.mutexes += 1;
        Handle(record.mutexes)
    }
    fn acquire(&self, _mutex: Handle, _timeout: u16) -> Result<(), AmlError> {
        Ok(())
    }
    fn release(&self, _mutex: Handle) {}
    fn handle_fatal_error(&self, fatal_type: u8, fatal_code: u32, fatal_arg: u64) {
        self.0.lock().unwrap().fatals.push((fatal_type, fatal_code, fatal_arg));
    }
    fn handle_notify(&self, path: &AmlName, value: u64) {
        self.0.lock().unwrap().notifies.push((path.as_string(), value));
    }
}

/// An embedded controller's 256 bytes, with every access recorded as (offset, width, written value).
/// One access: (offset, width in bits, the value written or `None` for a read).
pub type Access = (u64, u8, Option<u64>);

#[derive(Clone)]
pub struct FakeEc {
    pub space: Arc<Mutex<[u8; 256]>>,
    pub accesses: Arc<Mutex<Vec<Access>>>,
}

impl FakeEc {
    pub fn new() -> Self {
        FakeEc { space: Arc::new(Mutex::new([0; 256])), accesses: Arc::new(Mutex::new(Vec::new())) }
    }
}

impl RegionHandler for FakeEc {
    fn read(&self, _region: &OpRegion, offset: u64, width_bits: u8) -> Result<u64, AmlError> {
        self.accesses.lock().unwrap().push((offset, width_bits, None));
        let space = self.space.lock().unwrap();
        let mut value = 0u64;
        for i in 0..(width_bits as u64 / 8) {
            value |= (space[((offset + i) & 0xff) as usize] as u64) << (8 * i);
        }
        Ok(value)
    }
    fn write(&self, _region: &OpRegion, offset: u64, width_bits: u8, value: u64) -> Result<(), AmlError> {
        self.accesses.lock().unwrap().push((offset, width_bits, Some(value)));
        let mut space = self.space.lock().unwrap();
        for i in 0..(width_bits as u64 / 8) {
            space[((offset + i) & 0xff) as usize] = (value >> (8 * i)) as u8;
        }
        Ok(())
    }
}

pub fn interpreter() -> (Interpreter<TestHandler>, TestHandler) {
    let handler = TestHandler::default();
    (Interpreter::new_bare(handler.clone(), 2), handler)
}

/// An interpreter with small limits, so the limit tests run fast.
pub fn small_interpreter() -> (Interpreter<TestHandler>, TestHandler) {
    let (interpreter, handler) = interpreter();
    interpreter.set_limits(Limits {
        max_ops: 200_000,
        max_loop_iterations: 5_000,
        max_buffer_bytes: 64 * 1024,
        max_package_elements: 64,
        ..Limits::default()
    });
    (interpreter, handler)
}

pub fn name(path: &str) -> AmlName {
    path.parse().unwrap()
}

/// A PkgLength and its contents (the length counts its own bytes).
pub fn pkg(content: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let len = content.len();
    if len < 0x3f {
        out.push((len + 1) as u8);
    } else if len + 2 < (1 << 12) {
        let total = len + 2;
        out.push(0x40 | (total & 0xf) as u8);
        out.push((total >> 4) as u8);
    } else {
        let total = len + 3;
        out.push(0x80 | (total & 0xf) as u8);
        out.push((total >> 4) as u8);
        out.push((total >> 12) as u8);
    }
    out.extend_from_slice(content);
    out
}

/// A field element's or reserved field's PkgLength (a bare number of bits).
pub fn bits(count: usize) -> Vec<u8> {
    if count <= 0x3f { vec![count as u8] } else { vec![0x40 | (count & 0xf) as u8, (count >> 4) as u8] }
}

/// `Method(NAME, args) { body }`.
pub fn method(name: &[u8; 4], args: u8, body: &[u8]) -> Vec<u8> {
    let mut inner = name.to_vec();
    inner.push(args & 7);
    inner.extend_from_slice(body);
    let mut out = vec![0x14];
    out.extend(pkg(&inner));
    out
}

pub fn byte(value: u8) -> Vec<u8> {
    vec![0x0a, value]
}

pub fn dword(value: u32) -> Vec<u8> {
    let mut out = vec![0x0c];
    out.extend_from_slice(&value.to_le_bytes());
    out
}

pub fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// A tiny seeded generator (xorshift64*), so the random test is reproducible from its printed seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}
