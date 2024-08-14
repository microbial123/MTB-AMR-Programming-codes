pwd=$1;for i in `ls $1`;do a=`basename $i`;b=`ls $pwd/$a/*_1.fastq.gz`;c=`ls $pwd/$a/*_2.fastq.gz`;echo -e "$a\t$b\t$c";done > input.tab
source activate fastp
for d in `cat input.tab|awk '{print $1}'`;do e=`cat input.tab|grep $d|awk '{print $2}'`;f=`cat input.tab|grep $d|awk '{print $3}'`;fastp -i $e -I $f -o out.$d.1.fq.gz -O out.$d.2.fq.gz -j $d.json -h $d.html -w $3;done
conda deactivate
mkdir ./report
cp *.json ./report
source activate bowtie2
bowtie2-build $2 $(pwd)/MTB_reference
for d in `cat input.tab|awk '{print $1}'`;do e=`cat input.tab|grep $d|awk '{print $2}'`;f=`cat input.tab|grep $d|awk '{print $3}'`;bowtie2 -p $3 -x $(pwd)/MTB_reference -1 $e -2 $f | samtools sort -O bam -@ $3 -o - > $(pwd)/$d.bam;done
conda deactivate
source activate snippy
snippy-multi input.tab --ref $2 --cpus $3 > multi.sh
sh multi.sh
conda deactivate
for d in `cat input.tab|awk '{print $1}'`;do cat $(pwd)/$d/ref.fa.fai |awk '{print $1"\t1\t"$2}' > $d.bed;bamdst -p $d.bed -o ./$d $d.bam;done
source activate pandas
find -name "coverage.report" > coverage
python mergebamdst.py coverage coverage.xlsx
conda deactivate