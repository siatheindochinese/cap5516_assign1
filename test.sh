echo 'Evaluating resnet18 trained from ImageNet1K pretrained weights, with all augmentations'
python test.py --head resnet18 --ckpt pretrained_resnet18_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet18 trained from ImageNet1K pretrained weights, with vertical flip only'
python test.py --head resnet18 --ckpt pretrained_resnet18_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet18 trained from ImageNet1K pretrained weights, no augmentations'
python test.py --head resnet18 --ckpt pretrained_resnet18_xflip_0_yflip_0_rot_0.pt --gpu
echo 'Evaluating resnet34 trained from ImageNet1K pretrained weights, with all augmentations'
python test.py --head resnet34 --ckpt pretrained_resnet34_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet34 trained from ImageNet1K pretrained weights, with vertical flip only'
python test.py --head resnet34 --ckpt pretrained_resnet34_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet34 trained from ImageNet1K pretrained weights, no augmentations'
python test.py --head resnet34 --ckpt pretrained_resnet34_xflip_0_yflip_0_rot_0.pt --gpu
echo 'Evaluating resnet50 trained from ImageNet1K pretrained weights, with all augmentations'
python test.py --head resnet50 --ckpt pretrained_resnet50_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet50 trained from ImageNet1K pretrained weights, with vertical flip only'
python test.py --head resnet50 --ckpt pretrained_resnet50_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet50 trained from ImageNet1K pretrained weights, no augmentations'
python test.py --head resnet50 --ckpt pretrained_resnet50_xflip_0_yflip_0_rot_0.pt --gpu
echo 'Evaluating resnet18 trained from scratch, with all augmentations'
python test.py --head resnet18 --ckpt resnet18_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet18 trained from scratch, with vertical flip only'
python test.py --head resnet18 --ckpt resnet18_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet18 trained from scratch, no augmentations'
python test.py --head resnet18 --ckpt resnet18_xflip_0_yflip_0_rot_0.pt --gpu
echo 'Evaluating resnet34 trained from scratch, with all augmentations'
python test.py --head resnet34 --ckpt resnet34_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet34 trained from scratch, with vertical flip only'
python test.py --head resnet34 --ckpt resnet34_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet34 trained from scratch, no augmentations'
python test.py --head resnet34 --ckpt resnet34_xflip_0_yflip_0_rot_0.pt --gpu
echo 'Evaluating resnet50 trained from scratch, with all augmentations'
python test.py --head resnet50 --ckpt resnet50_xflip_0.5_yflip_0.5_rot_360.pt --gpu
echo 'Evaluating resnet50 trained from scratch, with vertical flip only'
python test.py --head resnet50 --ckpt resnet50_xflip_0_yflip_0.5_rot_0.pt --gpu
echo 'Evaluating resnet50 trained from scratch, no augmentations'
python test.py --head resnet50 --ckpt resnet50_xflip_0_yflip_0_rot_0.pt --gpu
