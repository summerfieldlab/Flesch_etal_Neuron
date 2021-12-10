# !/bin/bash
#
# tidies up fMRI data
# from granada testing sessions
#
# Timo Flesch

# ---------------------------------------
# Subjects 1-13
# set parameters
numRuns=6
runIDs=(8 12 16 20 24 28)


nameFunct="ep2d_64mx_3_5mm_TE_30ms_"
nameStruct="t1_mpr_sag_p2_iso_4"


nameFunctOut="functionalScan_run_"
nameStructOut="structuralScan"

# set IO directory names
inDir="`pwd`/raw"
outDir="`pwd`/renamed"

# do the magic.. for all subjects:
for ii in `ls  $inDir | grep -v TIMO014`;do
   # remove unneccessary subfolder (Mruz_Jan)
   mv $inDir/$ii/Mruz_Jan/* "$inDir/$ii/"
   rmdir $inDir/$ii/Mruz_Jan/
   # create subfloder for subject
   mkdir "$outDir/$ii"
   # copy subject's structural scan in subfolder with sensible name
   mkdir "$outDir/$ii/$nameStructOut"
   cp $inDir/$ii/${nameStruct}/* $outDir/$ii/$structDir/
   # do the same for each run (functional scan)
   for ((jj=0;jj<numRuns;jj++)); do
       functDir="$nameFunctOut`expr $jj + 1`"
       mkdir "$outDir/$ii/$functDir"
       cp $inDir/$ii/${nameFunct}${runIDs[jj]}/* $outDir/$ii/$functDir/
   done
done

# ------------------------------------------
# Subject 14
# set parameters
numRuns=6

runIDs=(10 14 18 22 26 30)

nameFunct="ep2d_64mx_3_5mm_TE_30ms_"
nameStruct="t1_mpr_sag_p2_iso_6"

nameFunctOut="functionalScan_run_"
nameStructOut="structuralScan"

# set IO directory names
inDir="`pwd`/raw"
outDir="`pwd`/renamed"

# do the magic.. for all subjects:
for ii in `ls  $inDir | grep TIMO014`;do
   # remove unneccessary subfolder (Mruz_Jan)
   mv $inDir/$ii/Mruz_Jan/* "$inDir/$ii/"
   rmdir $inDir/$ii/Mruz_Jan/
   # create subfloder for subject
   mkdir "$outDir/$ii"
   # copy subject's structural scan in subfolder with sensible name
   mkdir "$outDir/$ii/$nameStructOut"
   cp $inDir/$ii/${nameStruct}/* $outDir/$ii/$structDir/
   # do the same for each run (functional scan)
   for ((jj=0;jj<numRuns;jj++)); do
       functDir="$nameFunctOut`expr $jj + 1`"
       mkdir "$outDir/$ii/$functDir"
       cp $inDir/$ii/${nameFunct}${runIDs[jj]}/* $outDir/$ii/$functDir/
   done
done
