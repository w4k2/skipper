
# Hyperplane 

-i instanceRandomSeed (default: 1)
Seed for random generation of instances.
-c numClasses (default: 2)
The number of classes to generate.
-a numAtts (default: 10)
The number of attributes to generate.
-k numDriftAtts (default: 2)
The number of attributes with drift.
-t magChange (default: 0.0)
Magnitude of the change for every example
-n noisePercentage (default: 5)
Percentage of noise to add to the data.
-s sigmaPercentage (default: 10)
Percentage of probability that the direction of change is reversed.


```
java -cp lib/moa.jar -javaagent:lib/sizeofag-1.0.4.jar moa.DoTask "WriteStreamToARFFFile -s (generators.HyperplaneGenerator -i 2 -a 32 -k 8 -t 0.29457 -n 6 -s 11) -f ./moa_streams/raw/hyperplane.arff -m 50000"
```