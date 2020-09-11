#!/bin/bash -ex
source /opt/Xilinx/Vivado/2019.2/settings64.sh
source /opt/Xilinx/Vitis/2019.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
source /opt/petalinux/settings.sh

mkdir ultra96v2_oob
cd ultra96v2_oob

git clone -b 2019.2 https://github.com/Xilinx/Vitis_Embedded_Platform_Source.git
cp -rf ./Vitis_Embedded_Platform_Source/Xilinx_Official_Platforms/zcu102_base/* .
mkdir hdl
mkdir petalinux_oob
mkdir bdf
mkdir download
cd download
#download Git
git clone -b 2019.1 https://github.com/Avnet/petalinux.git
git clone -b 2019.1 https://github.com/Avnet/hdl.git
#make work directory
mkdir ../hdl/Boards
mkdir ../hdl/IP
mkdir ../hdl/Projects
mkdir ../hdl/Scripts
mkdir ../hdl/Scripts/ProjectScripts
mkdir ../petalinux_oob/scripts
mkdir ../petalinux_oob/configs
mkdir ../petalinux_oob/configs/device-tree
mkdir ../petalinux_oob/configs/kernel
mkdir ../petalinux_oob/configs/meta-user
mkdir ../petalinux_oob/configs/project
mkdir ../petalinux_oob/configs/rootfs
mkdir ../petalinux_oob/configs/u-boot

#copy files
cp -rf hdl/Boards/ULTRA96V2 ../hdl/Boards
cp -rf hdl/IP/PWM_w_Int ../hdl/IP
cp -rf hdl/Projects/ultra96v2_oob ../hdl/Projects
cp hdl/Scripts/make_ultra96v2_oob.tcl ../hdl/Scripts
cp hdl/Scripts/make.tcl ../hdl/Scripts
cp hdl/Scripts/bin_helper.tcl ../hdl/Scripts
cp hdl/Scripts/ProjectScripts/ultra96v2_oob.tcl ../hdl/Scripts/ProjectScripts
cp hdl/Scripts/tag.tcl ../hdl/Scripts

cp petalinux/scripts/make_ultra96v2_oob_bsp.sh ../petalinux_oob/scripts
cp petalinux/configs/device-tree/system-user.dtsi.ULTRA96V2 ../petalinux_oob/configs/device-tree
cp petalinux/configs/kernel/user.cfg.ULTRA96V2 ../petalinux_oob/configs/kernel
cp -rf petalinux/configs/meta-user/ultra96v2_oob ../petalinux_oob/configs/meta-user
cp petalinux/configs/project/config.ultra96v2_oob.patch ../petalinux_oob/configs/project
cp petalinux/configs/project/config.sd_ext4_boot.patch ../petalinux_oob/configs/project
cp petalinux/configs/rootfs/config.ultra96v2_oob ../petalinux_oob/configs/rootfs
cp petalinux/configs/u-boot/platform-top.h.ultra96v2_sd_boot ../petalinux_oob/configs/u-boot
cp petalinux/configs/u-boot/bsp.cfg ../petalinux_oob/configs/u-boot

cd ..

#change vivado 
cd vivado
mv zcu102_base_xsa.tcl ultra96v2_oob_xsa.tcl.bak

cat ultra96v2_oob_xsa.tcl.bak | sed -e 's/zcu102_base/ultra96v2_oob/g' -e 's/zcu102/ultra96/g' > ultra96v2_oob_xsa.tcl1.bak
cat ultra96v2_oob_xsa.tcl1.bak | sed -e '43i   source ../hdl/Boards/ULTRA96V2/ultra96v2_oob.tcl -notrace' | sed -e '44i avnet_create_project ultra96v2_oob ultra96v2_oob Project' | sed -e '45i set_property board_part em.avnet.com:ultra96v2:part0:1.0 [current_project]' | sed -e '47,51d' > ultra96v2_oob_xsa.tcl2.bak
cat ultra96v2_oob_xsa.tcl2.bak | sed -e '47i set_property ip_repo_paths  ../hdl/IP [current_fileset]' | sed -e '48i update_ip_catalog' | sed -e '388,1890d' | sed -e '388i  avnet_add_ps_preset ultra96v2_oob ultra96v2_oob ultra96v2_oob' |  sed -e '389i set_property name ps_e [get_bd_cells zynq_ultra_ps_e_0]' |  sed -e '431i avnet_add_user_io_preset ultra96v2_oob ultra96v2_oob ultra96v2_oob' > ultra96v2_oob_xsa.tcl3.bak
cat ultra96v2_oob_xsa.tcl3.bak | sed -e '439s/set i 1/set i 10/g' | sed -e '469i add_files -fileset constrs_1 -norecurse ../hdl/Projects/ultra96v2_oob/ultra96v2_oob.xdc'| sed -e '470i import_files -fileset constrs_1 ../hdl/Projects/ultra96v2_oob/ultra96v2_oob.xdc' > ultra96v2_oob_xsa.tcl4.bak
cat ultra96v2_oob_xsa.tcl4.bak | sed -e 's/-jobs 16/-jobs 6/g' > ultra96v2_oob_xsa.tcl
rm *.bak

#Chang directory for script
cd ..
cd hdl/Boards/ULTRA96V2
mv ultra96v2_oob.tcl ultra96v2_oob.tcl.bak
cat ultra96v2_oob.tcl.bak | sed -e '68,71d' | sed -e '442,444d' | sed -e '479i startgroup' | sed  -e '480i set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ultra_ps_e_0]' | sed   -e '481i set_property -dict [list CONFIG.PSU__USE__M_AXI_GP2 {1}] [get_bd_cells zynq_ultra_ps_e_0]' | sed -e '482i set_property -dict [list CONFIG.PSU__USE__S_AXI_GP5 {1}] [get_bd_cells zynq_ultra_ps_e_0]' | sed -e  '483i endgroup' > ultra96v2_oob.tcl1.bak
cat ultra96v2_oob.tcl1.bak | sed -e  '462,464d' | sed -e '61,77d' | sed -e '61i delete_bd_objs [get_bd_nets axi_intc_0_irq]' | sed -e '62i startgroup' | sed -e '63i create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0' | sed -e '64i endgroup' | sed -e '65i set_property -dict [list CONFIG.NUM_PORTS {1}] [get_bd_cells xlconcat_0]' | sed -e '66i connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins ps_e/pl_ps_irq0]' | sed -e '67i connect_bd_net [get_bd_pins axi_intc_0/irq] [get_bd_pins xlconcat_0/In0]' > ultra96v2_oob.tcl2.bak
cat ultra96v2_oob.tcl2.bak | sed -e  's/zynq_ultra_ps_e_0\/pl_clk0 (100 MHz)/clk_wiz_0\/clk_out3 (75 MHz)/g' -e  's/zynq_ultra_ps_e_0\/M_AXI_HPM0_FPD/ps_e\/M_AXI_HPM0_LPD/g' -e 's/ps8_0_axi_periph/interconnect_axilite/g' > ultra96v2_oob.tcl3.bak
cat ultra96v2_oob.tcl3.bak | sed -e '243,249d' | sed -e '244d' | sed -e 's/CONFIG.NUM_PORTS {5}/CONFIG.NUM_PORTS {6}/g' -e 's/xlconcat_0\/In4/xlconcat_0\/In5/g' -e 's/xlconcat_0\/In3/xlconcat_0\/In4/g' -e 's/xlconcat_0\/In2/xlconcat_0\/In3/g' -e 's/xlconcat_0\/In1/xlconcat_0\/In2/g' -e '240,250s/xlconcat_0\/In0/xlconcat_0\/In1/g' | sed -e '243,424s/zynq_ultra_ps_e_0/ps_e/g' > ultra96v2_oob.tcl4.bak
mv ultra96v2_oob.tcl4.bak ultra96v2_oob.tcl
rm *.bak

cd ../../../..

#cp -rf vivado ultra96v2_oob
cd ultra96v2_oob/vivado
make PLATFORM=ultra96v2_oob XILINX_VIVADO=/opt/Xilinx/Vivado/2019.2
