#!/bin/bash -ex
source /opt/Xilinx/Vivado/2019.2/settings64.sh
source /opt/Xilinx/Vitis/2019.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
source /opt/petalinux/settings.sh

cp ultra96v2_oob/vivado/ultra96v2_oob.xsa ultra96v2_oob/petalinux

cd ultra96v2_oob/petalinux
make refresh_hw XSA_DIR=../vivado

cd project-spec/configs
mv config config.bak
mv rootfs_config rootfs_config.bak

cat config.bak | sed -e 's/PSU_UART_0_/PSU_UART_1_/g' -e 's/# CONFIG_SUBSYSTEM_SERIAL_PSU_UART_1_SELECT is not set/# CONFIG_SUBSYSTEM_SERIAL_PSU_UART_0_SELECT is not set/g' -e 's/psu_uart_0/psu_uart_1/g' -e 's/cadence/cadence1/g' > config1.bak
cat config1.bak | sed -e 's/zcu102-rev1.0/avnet-ultra96-rev1/g' -e 's/xilinx_zynqmp_zcu102_rev1_0_defconfig/avnet_ultra96_rev1_defconfig/g' -e 's/xilinx-zcu102/ultra96v2-oob/g' -e 's/zcu102-zynqmp/ultra96-zynqmp/g' -e 's/CONFIG_SUBSYSTEM_PRIMARY_SD_PSU_SD_1_SELECT=y/CONFIG_SUBSYSTEM_PRIMARY_SD_PSU_SD_0_SELECT=y/g' -e 's/# CONFIG_SUBSYSTEM_PRIMARY_SD_PSU_SD_0_SELECT is not set/# CONFIG_SUBSYSTEM_PRIMARY_SD_PSU_SD_1_SELECT is not set/g' > config2.bak
cat config2.bak | sed -e 's/CONFIG_SUBSYSTEM_ROOTFS_INITRAMFS=y/# CONFIG_SUBSYSTEM_ROOTFS_INITRAMFS is not set/g' -e 's/# CONFIG_SUBSYSTEM_ROOTFS_EXT is not set/CONFIG_SUBSYSTEM_ROOTFS_EXT=y/g' -e 's/115200 clk_ignore_unused/115200 clk_ignore_unused root=\/dev\/mmcblk0p2 rw rootwait/g' | sed -e '182i CONFIG_SUBSYSTEM_SDROOT_DEV="/dev/mmcblk0p2"' > config3.bak
cat rootfs_config.bak | sed -e 's/# CONFIG_bc is not set/CONFIG_bc=y/g' -e 's/# CONFIG_i2c-tools is not set/CONFIG_i2c-tools=y/g' -e 's/# CONFIG_usbutils is not set/CONFIG_usbutils=y/g' -e 's/# CONFIG_ethtool is not set/CONFIG_ethtool=y/g' -e  's/# CONFIG_git is not set/CONFIG_git=y/g' > rootfs_config1.bak
cat rootfs_config1.bak | sed -e 's/# CONFIG_coreutils is not set/CONFIG_coreutils=y/g' -e 's/# CONFIG_openamp-fw-echo-testd is not set/CONFIG_openamp-fw-echo-testd=y/g' -e 's/# CONFIG_openamp-fw-mat-muld is not set/CONFIG_openamp-fw-mat-muld=y/g' -e 's/# CONFIG_openamp-fw-rpc-demo is not set/CONFIG_openamp-fw-rpc-demo=y/g' > rootfs_config2.bak
cat rootfs_config2.bak | sed -e 's/# CONFIG_packagegroup-petalinux is not set/CONFIG_packagegroup-petalinux=y/g' -e 's/# CONFIG_packagegroup-petalinux-benchmarks is not set/CONFIG_packagegroup-petalinux-benchmarks=y/g' -e 's/# CONFIG_packagegroup-petalinux-matchbox is not set/CONFIG_packagegroup-petalinux-matchbox=y/g' > rootfs_config3.bak
cat rootfs_config3.bak | sed -e 's/# CONFIG_packagegroup-petalinux-openamp is not set/CONFIG_packagegroup-petalinux-openamp=y/g' -e 's/# CONFIG_packagegroup-petalinux-self-hosted is not set/CONFIG_packagegroup-petalinux-self-hosted=y/g' -e 's/# CONFIG_packagegroup-petalinux-utils is not set/CONFIG_packagegroup-petalinux-utils=y/g' -e 's/# CONFIG_packagegroup-petalinux-v4lutils is not set/CONFIG_packagegroup-petalinux-v4lutils=y/g' -e 's/# CONFIG_packagegroup-petalinux-x11 is not set/CONFIG_packagegroup-petalinux-x11=y/g' -e 's/# CONFIG_imagefeature-package-management is not set/CONFIG_imagefeature-package-management=y/g' > rootfs_config4.bak
cat rootfs_config4.bak | sed -e '$a CONFIG_wilc=y' -e '$a CONFIG_libftdi=y' -e '$a CONFIG_bonniePLUSPLUS=y' -e '$a CONFIG_cmake=y' -e '$a CONFIG_iperf3=y' -e '$a CONFIG_iw=y' -e '$a CONFIG_lmsensors-sensorsdetect=y' -e '$a CONFIG_nano=y' -e '$a CONFIG_packagegroup-base-extended=y' -e '$a CONFIG_packagegroup-petalinux-96boards-sensors=y' -e '$a CONFIG_packagegroup-petalinux-ultra96-webapp=y' -e '$a CONFIG_python-pyserial=y' -e '$a CONFIG_python3-pip=y' -e '$a CONFIG_ultra96-ap-setup=y' -e '$a CONFIG_ultra96-misc=y' -e '$a CONFIG_wilc-firmware-wilc3000=y' -e '$a CONFIG_ultra96-radio-leds=y' -e '$a CONFIG_ultra96-wpa=y' -e '$a CONFIG_sds-lib=y' -e '$a CONFIG_wilc3000-fw=y'  -e '$a CONFIG_ultra96-startup-pages=y'  > rootfs_config5.bak
cat rootfs_config5.bak | sed -e 's/# CONFIG_packagegroup-petalinux-weston is not set/CONFIG_packagegroup-petalinux-weston=y/g' > rootfs_config6.bak
mv config3.bak config
mv rootfs_config6.bak rootfs_config
rm *.bak

cd ..
#meta-user/recipes-kernel/linux/linux-xlnx/user_2019-10-31-20-33-00.cfg

cd meta-user/recipes-bsp/device-tree/
cp ../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/device-tree.bbappend .
cd files
cp -rf ../../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/files/multi-arch .
cat system-user.dtsi | sed -e '1,25d' > system-user.dtsi.bak
cat ../../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/files/system-user.dtsi system-user.dtsi.bak > system-user.dtsi
cat openamp.dtsi | sed -e '1,42d' > openamp.dtsi.bak
cat ../../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/files/openamp.dtsi | sed -e '56,57d' > openamp.dtsi1.bak
cat openamp.dtsi1.bak openamp.dtsi.bak > openamp.dtsi
cp ../../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/files/xen.dtsi .
cp ../../../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/device-tree/files/zynqmp-qemu-arm.dts .
rm *.bak

cd ../../..

#meta-user copy
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-core .
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-graphics .
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-modules .
mkdir -p recipes-utils 
cd recipes-utils 
cp -rf ../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-utils/ultra96-radio-leds .
cp -rf ../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-utils/ultra96-wpa .
cp -rf ../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-utils/ultra96-ap-setup .
cd ..
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-apps/sds-lib recipes-apps
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/u-boot recipes-bsp
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/pmu-firmware recipes-bsp
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/ultra96-misc recipes-bsp
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-bsp/wilc3000-fw recipes-bsp
cp -rf ../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-kernel .

cd conf
mv petalinuxbsp.conf petalinuxbsp.conf.bak
mv user-rootfsconfig user-rootfsconfig.bak 
mv ../../../../petalinux_oob/configs/meta-user/ultra96v2_oob/recipes-core/images/petalinux-image-full.bbappend petalinux-image-full.bbappend.bak
cat petalinuxbsp.conf.bak | sed -e '$a MACHINE_FEATURES_remove_ultra96-zynqmp = "mipi"' -e '$a DISTRO_FEATURES_append = " bluez5 dbus"' -e '$a PREFERRED_VERSION_wilc-firmware = "15.2" ' > petalinuxbsp.conf
cat petalinux-image-full.bbappend.bak | sed -e '3,4d' | sed -e '/#/d' | sed -e "/^[<space><tab>]*$/d" | sed -e 's/IMAGE_INSTALL_append = " /CONFIG_/g' -e 's/"//g' > user-rootfsconfig1.bak
cat user-rootfsconfig.bak user-rootfsconfig1.bak | sed  -e '$a CONFIG_ultra96-startup-pages'  > user-rootfsconfig
rm *.bak
cd ..

cd recipes-kernel/linux

mv linux-xlnx_%.bbappend linux-xlnx_%.bbappend.bak
cat  linux-xlnx_%.bbappend.bak | sed -e '3iSRC_URI += "file://user.cfg"' > linux-xlnx_%.bbappend
rm *.bak

cd ../..

cd recipes-kernel/linux/linux-xlnx
mv user_2019-10-31-20-33-00.cfg user_2019-10-31-20-33-00.cfg.bak
cat user_2019-10-31-20-33-00.cfg.bak | sed -e 's/CONFIG_CMA_SIZE_MBYTES=1024/CONFIG_CMA_SIZE_MBYTES=512/g'  > user_2019-10-31-20-33-00.cfg
cd ../../../../..

make all XSA_DIR=../vivado PLATFORM=ultra96v2_oob
make sysroot
cd ..

mv Makefile Makefile.bak
cat Makefile.bak | sed -e 's/zcu102_base/ultra96v2_oob/g' > Makefile
rm *.bak
mv scripts/zcu102_base_pfm.tcl scripts/ultra96v2_oob_pfm.tcl
make pfm

cd ..

if [ ! -d sd_card ]; then
    mkdir sd_card
    mkdir sd_card/boot 
    mkdir sd_card/rootfs
fi
cp ultra96v2_oob/petalinux/images/linux/image.ub ultra96v2_oob/petalinux/images/linux/BOOT.BIN ultra96v2_oob/petalinux/images/linux/system.dtb sd_card/boot
cp ultra96v2_oob/petalinux/images/linux/rootfs.tar.gz sd_card/rootfs 
