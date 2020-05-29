## Datasets

### Clinical Trials Dataset
This dataset contains international clinical trials record from different countries. We used a portion of this dataset that includes data from four countries: Canada, Germany, Italy and Japan. It is available from [source](https://old.datahub.io/dataset/linkedct). It has 29 attributes and describes patient demographics, diagnosis, symptoms, condition, etc.

**Attribute Schema**   
```
id
facility_address_country
download_date
org_study_id
nct_id
brief_title
acronym
official_title
lead_sponsor_agency
source
overall_status
why_stopped
phase
study_type
study_design
number_of_arms
diagnosis
enrollment
biospec_retention
eligibility_sampling_method
eligibility_gender
eligibility_minimum_age
eligibility_maximum_age
eligibility_healthy_volunteers
condition
measure
time_frame
safety_issue
drug_name
```

**Functional Dependencies:**
age, overall_status, diagnosis -> drug_name
overall_status, time_frame, measure -> condition


### Census Dataset 
The dataset is from the U.S. Census Bureau [source](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.html). It has 40 attributes and provides population characteristics such as education level, years of schooling, occupation, income, and age, etc.

**Attribute Schema**   
```
age AAGE
class of worker ACLSWKR
industry code ADTIND
occupation ADTOCC
adjusted gross income AGI
education AEDU
education-num AEDU-NUM
wage per hour AHRSPAY
enrolled in edu inst last wk AHSCOL
marital status AMARITL
major industry code AMJIND
major occupation code AMJOCC
race ARACE
hispanic Origin AREORGN
sex ASEX
member of a labor union AUNMEM
reason for unemployment AUNTYPE
full or part-time employment stat AWKSTAT
capital gains CAPGAIN
capital losses CAPLOSS
dividends from stocks DIVVAL
federal income tax liability FEDTAX
tax filer status FILESTAT
region of previous residence GRINREG
state of previous residence GRINST
detailed household and family stat HHDFMX
detailed household summary in household HHDREL
instance weight MARSUPWT
migration code-change in msa MIGMTR1
migration code-change in reg MIGMTR3
migration code-move within reg MIGMTR4
live in this house 1 year ago MIGSAME
migration prev res in sunbelt MIGSUN
num persons worked for employer NOEMP
family members under 18 PARENT
total person earnings PEARNVAL
country of birth father PEFNTVTY
country of birth mother PEMNTVTY
country of birth self PENATVTY
citizenship PRCITSHP
total person income PTOTVAL
own business or self-employed SEOTR
taxable income amount TAXINC
fill inc questionnaire for veteran's admin VETQVA
veterans benefit VETYN
weeks worked in year WKSWORK
```

**Functional Dependencies:**
 age, education-num -> education
 age, industry code, occupation -> wage-per-hour 


### Food Inspection
This dataset is from NYU open data [source](https://opendata.cityofnewyork.us/), which  has 11 attributs and provides violation citations of inspected restaurants in New York City. It contains the restaurant address, violation code, violation description, zipcode, etc.

**Attribute Schema**

```
borough, address, violation code, violation description, zipcode, cuisine
description, action, inspection type, critical flag, score, grade
```

**Functional Dependencies:**
borough, zipcode -> address
violation code, inspection type -> violation description


## Source Code
The source code is available [here](https://github.com/PrivacyPreversingDataCleaning/Privacy-Aware-Data-Cleaning-as-a-Service).
