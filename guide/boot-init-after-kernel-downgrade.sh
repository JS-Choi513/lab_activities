#!/bin/bash

#아래 두 코드는 주석 처리하거나 삭제할 것
echo "please read the file before execution"
exit 1


#vmlinuz, initrd.img, grub이 최신 커널 버전으로 되어있는 것을 강제로 다운그레이드 하는 방법
#기존 커널 버전의 관련 파일들을 모두 삭제하고 vmlinuz, initrd.img, grub을 다시 세팅

#다운그레이드한 커널 버전
#config-{버전 명}, initrd.img-{버전 명}에 들어가는 버전 명을 아래에 넣을 것
#$version 이외의 버전의 설정 파일은 다 삭제 예정
version="5.15.21"

#initrd.img 관련 파일 전부 삭제
rm `ls /boot/initrd.img* | grep -v $version`
#initrd.img 심볼릭 링크 생성
ln -s /boot/initrd.img-$version /boot/initrd.img

#vmlinuz 관련 파일 전부 삭제
rm `ls /boot/vmlinuz* | grep -v $version`
#vmlinuz 심볼릭 링크 생성
ln -s /boot/vmliuz-$version /boot/vmlinuz

#grub config파일 재설정
#grub-mkconfig -o {grub.cfg 위치}
#grub.cfg의 위치의 경우 기존 grub.cfg 파일 위치를 확인 후 그곳으로 설정
grub-mkconfig -o /boot/grub/grub.cfg

#커널 버전에 따라 각 파일들의 위치가 다를 수 있으니 필수적으로 확인 후 실행할 것
