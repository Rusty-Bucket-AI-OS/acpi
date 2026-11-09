use crate::aml::{AmlError, namespace::AmlName};

#[derive(Clone, Debug)]
pub struct OpRegion {
    pub space: RegionSpace,
    pub base: u64,
    pub length: u64,
    pub parent_device_path: AmlName,
}

/// Hardening: a host's handler for one region space (the embedded controller's above all). The interpreter splits a
/// field into accesses of the field's access width (a `ByteAcc` field of 16 bits is two 8-bit reads, composed
/// little-endian), checks each against the region's length, and calls the handler with the lock released.
pub trait RegionHandler: Send + Sync {
    /// Reads `width_bits` (8, 16, 32 or 64) at byte `offset` from the start of `region`.
    fn read(&self, region: &OpRegion, offset: u64, width_bits: u8) -> Result<u64, AmlError>;
    /// Writes the low `width_bits` of `value` at byte `offset` from the start of `region`.
    fn write(&self, region: &OpRegion, offset: u64, width_bits: u8, value: u64) -> Result<(), AmlError>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum RegionSpace {
    SystemMemory,
    SystemIO,
    PciConfig,
    EmbeddedControl,
    SmBus,
    SystemCmos,
    PciBarTarget,
    Ipmi,
    GeneralPurposeIo,
    GenericSerialBus,
    Pcc,
    Oem(u8),
}

impl From<u8> for RegionSpace {
    fn from(value: u8) -> Self {
        match value {
            0 => RegionSpace::SystemMemory,
            1 => RegionSpace::SystemIO,
            2 => RegionSpace::PciConfig,
            3 => RegionSpace::EmbeddedControl,
            4 => RegionSpace::SmBus,
            5 => RegionSpace::SystemCmos,
            6 => RegionSpace::PciBarTarget,
            7 => RegionSpace::Ipmi,
            8 => RegionSpace::GeneralPurposeIo,
            9 => RegionSpace::GenericSerialBus,
            10 => RegionSpace::Pcc,
            _ => RegionSpace::Oem(value),
        }
    }
}

// Hardening: the space's number, as `_REG` takes it.
impl From<RegionSpace> for u8 {
    fn from(space: RegionSpace) -> u8 {
        match space {
            RegionSpace::SystemMemory => 0,
            RegionSpace::SystemIO => 1,
            RegionSpace::PciConfig => 2,
            RegionSpace::EmbeddedControl => 3,
            RegionSpace::SmBus => 4,
            RegionSpace::SystemCmos => 5,
            RegionSpace::PciBarTarget => 6,
            RegionSpace::Ipmi => 7,
            RegionSpace::GeneralPurposeIo => 8,
            RegionSpace::GenericSerialBus => 9,
            RegionSpace::Pcc => 10,
            RegionSpace::Oem(value) => value,
        }
    }
}
