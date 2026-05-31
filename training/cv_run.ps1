Set-Location '<X2CT_ROOT>'
$py = "python"
$yml = "./experiment/real_singleview_segMC/real_singleview_segMC.yml"
for ($k = 1; $k -le 5; $k++) {
  $tag = "segMC_fold$k"
  $ckpt = "save_models/singleView_CTGAN/LIDC256_segMC/$tag/checkpoint/80"
  if (-not (Test-Path $ckpt)) {
    Write-Output "=== TRAIN fold $k ($(Get-Date -Format HH:mm)) ==="
    & $py -u train.py --ymlpath=$yml --gpu=0 --dataroot=./data/LIDC256_segMC --dataset=train --tag=$tag --data=LIDC256_segMC --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/cv_train_fold$k.txt --valid_datasetfile=./data/cv_test_fold$k.txt --valid_dataset=test *> "cv_train_fold$k.log"
  }
  $rd = "cv_fold${k}_e80"
  if (Test-Path $rd) { Remove-Item $rd -Recurse -Force }
  Write-Output "=== INFER fold $k e80 ($(Get-Date -Format HH:mm)) ==="
  & $py visual.py --ymlpath=$yml --gpu=0 --dataroot=./data/LIDC256_segMC --dataset=test --tag=$tag --data=LIDC256_segMC --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/cv_test_fold$k.txt --resultdir="./$rd" --check_point=80 --how_many=30 *> "cv_infer_fold$k.log"
  Write-Output "=== fold $k DONE ($(Get-Date -Format HH:mm)) ==="
}
Write-Output "ALL FOLDS COMPLETE"
