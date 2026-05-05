############## NET - IOSTANDARD #####################
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
#############SPI Configurate Setting##################
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design] 
set_property CONFIG_MODE SPIx4 [current_design] 
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design] 
############## board clock define ####################
# FACE_K7_SVIC_V10_250916.pdf, sheet 2/11:
# FPGA_SYS_CLK_50M -> U27 (Bank 14, 3.3V)
create_clock -period 20.000 [get_ports sys_clk]
set_property PACKAGE_PIN U27 [get_ports sys_clk]
set_property IOSTANDARD LVCMOS33 [get_ports sys_clk]
############## usb uart define #######################
# FACE_K7_SVIC_V10_250916.pdf, sheet 2/11:
# FDT0_FPGA_TXD -> W22 (FT2232H TXD, FPGA receives)
# FDT0_FPGA_RXD -> W21 (FT2232H RXD, FPGA transmits)
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property PACKAGE_PIN W22 [get_ports uart_rx]

set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property PACKAGE_PIN W21 [get_ports uart_tx]
