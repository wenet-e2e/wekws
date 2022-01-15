#!/bin/bash
# trim silence of keywords wav files at both sides, while fillers are left intact

nj=8
. tools/parse_options.sh || exit 1;

inscp=$1
text=$2 # text files that map wav ids to labels
outdir=$3
indir=$(dirname ${inscp})

if [ $# -eq 4 ]; then
  logdir=$4
else
  logdir=${indir}/log
fi

mkdir -p ${logdir}
rm -f $logdir/wav_*.slice
rm -f $logdir/wav_*.shape

pasted_scp=$logdir/pasted_scp.tmp
paste -d ' ' $inscp $text > $pasted_scp
split --additional-suffix .slice -d -n l/$nj $pasted_scp $logdir/wav_
rm -f $pasted_scp

if [ -n "$outdir" ]; then
  echo "trimming silence in <$inscp>"
  mkdir -p $outdir
fi

trim_dir=$(realpath $outdir)
trim_count=0
total_count=0
# trimmed empty files are logged to this scp
empty_scp=$indir/empty_wav.scp
echo -n '' > $empty_scp
trim_scp=$indir/trim_wav.scp
echo -n '' > $trim_scp

shopt -s nullglob
for slice in "$logdir"/wav_*.slice; do
{
  while read -r line; do
    IFS=' ' read -ra arr <<< "$line"
    if [ ${#arr[@]} -ne 4 ] || [ "${arr[0]}" != "${arr[2]}" ]; then
      echo "the \"paste($1 $2)\" expects 4 entries at each line: \"id1 path id2 label\" (id1 == id2)"
      exit 1
    fi
    wav_id=${arr[0]}
    wav_file=$(basename ${arr[1]})
    label=${arr[3]}
    # assume fillers' labels are -1, keywords' labels are positive integers
    if [ "$label" -gt "-1" ]; then
      # trim silence, which is less than 1% amplitude, until encounters non-sil that is at least 0.1s
      # NOTE, silence is trimmed to be at most 50ms long (not completely removed)
      sox ${arr[1]} ${trim_dir}/${wav_file} silence -l 1 0.1 1% -1 0.05 1% reverse silence -l 1 0.1 1% -1 0.05 1% reverse
      fs=$(stat -c '%s' ${trim_dir}/${wav_file})
      if [ "$fs" -le "44" ]; then
        echo "${trim_dir}/${wav_file} is empty after silence-trimming, please check its validity"
        echo "$wav_id ${trim_dir}/${wav_file} $label" >> $empty_scp
      else
        echo "$wav_id ${trim_dir}/${wav_file} $label" >> $trim_scp
      fi
      trim_count=$(($trim_count+1))
    else
      # fillers are not trimmed and softly linked to save disk spaces
      [ ! -f ${trim_dir}/${wav_file} ] && ln -s ${arr[1]} ${trim_dir}/${wav_file}
      echo "$wav_id ${trim_dir}/${wav_file} $label" >> $trim_scp
    fi
    total_count=$((total_count+1))
    [ $(($total_count % 100)) -eq 0 ] && echo -ne "job id:$BASHPID is working, ${trim_count}/${total_count} wavs are trimmed\r"
  done < $slice
} &
done

wait
cp $inscp $indir/wav_before_trim.scp
echo "the previous <$inscp> is renamed to <wav-no-trim.scp>"
mv $trim_scp $indir/wav.scp
echo "the new <$indir/wav.scp> now refers to trimmed wav files"
echo "${trim_count}/${total_count} wav files are trimmed"
echo ""
