
# path to openSMILE 2.3 SMILExtract
openSMILE=C:\toolkits\opensmile-2.3.0\bin\Win32/SMILExtract

"$openSMILE" -C "$configFile" -appendarfflld 1 -instname "$wavefile" -I "$audio_dir/$wavefile" -lldarffoutput "$train_lld_codebook_arff" -arfftargetsfile arff_targets.conf.inc -class ?  # class label does not matter (unsupervised codebook generation)
