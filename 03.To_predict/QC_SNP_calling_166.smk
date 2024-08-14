from glob import glob
import os

fs = glob("/data/MTB_research/PMID34880835_166/166_fq/*1.fastq.gz")
rs = [i.replace("1.fastq.gz", "2.fastq.gz") for i in fs]
samples = [os.path.basename(i).replace("_1.fastq.gz", "") for i in fs]
samples_dict = dict(zip(samples, zip(fs, rs)))
ref = "/data/MTB_research/PMID34880835_166/genes.gbk"

def get_fastq(wildcards):
    return samples_dict[wildcards.sample]


rule all:
    input:
        expand("rst/{sample}", sample=samples),


rule fastp:
    input:
        get_fastq,
    output:
        r1="clean_data/{sample}_1.fastq.gz",
        r2="clean_data/{sample}_2.fastq.gz",
        html="clean_data/{sample}.html",
        json="clean_data/{sample}.json",
    threads: 8
    shell:
        """

        fastp -i {input[0]} -I {input[1]} -o {output.r1} -O {output.r2} -h {output.html} -j {output.json} -w {threads} -q 20 -n 0 -l 36

        """


rule snippy:
    input:
        r1=rules.fastp.output.r1,
        r2=rules.fastp.output.r2,
    output:
        directory("rst/{sample}"),
    params:
        ref=ref,
    threads: 16
    shell:
        """

        snippy --cpus {threads} --outdir {output} --ref {params.ref} --R1 {input.r1} --R2 {input.r2}

        """

