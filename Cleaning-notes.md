
# Removals

* ipv4 src address
* ipv4 dst address
* dataset 
* no  row were dropped,no null or empty values

 # Observations on data
* The date set was pre cleaned, 4 datatypes were objects so i dropped 3/4 of the columns. the attck column was converted to int64 assigning each attack a numerical valuu and the attack is string for visualization purposes.


# Stats 
Total Entries:11994893
Attacks:9208048
Benign:2786845   
{'Benign': 0, 'Exploits': 1, 'Reconnaissance': 2, 'DoS': 3, 'Generic': 4, 'Shellcode': 5, 'Backdoor': 6, 'Fuzzers': 7, 'Worms': 8, 'Analysis': 9, 'injection': 10, 'DDoS': 11, 'scanning': 12, 'password': 13, 'mitm': 14, 'xss': 15, 'ransomware': 16, 'Infilteration': 17, 'Bot': 18, 'Brute Force': 19, 'Theft': 20}
Attack
Benign            9208048 0
DDoS               763285 11
Reconnaissance     482946 2
injection          468575 10
DoS                348962 3
Brute Force        291955 19
password           156299 13
xss                 99944 15
Infilteration       62072 17
Exploits            24736 1
scanning            21467 12
Fuzzers             19463 7
Backdoor            19029 6
Bot                 15683 18
Generic              5570 4
Analysis             1995 9
Theft                1909 20
Shellcode            1365 5
mitm                 1295 14
Worms                 153 8
ransomware            142 16
Name: count, dtype: int64

