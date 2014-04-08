Self-Organizing Maps
=========

Self-Organizing Maps is a form of machine learning technique which employs unsupervised learning. It means that you don't need to explicitly tell the SOM about what to learn in the input data. It automatically learns the patterns in input data and organizes the data into different groups.

How use
--------------

```sh
git clone https://github.com/jgabriellima/self_organization_map.git
python test.py
```


SOM on Step-by-Step
---------------

>1. Initialize weights with random values. 
>2. Present input pattern to the network. 
>3. Choose the output neuron with the highest activation state (Neurôno "winner"). 
>4. Update the weights of the winner neuron neighboring neurons, using a factor of learning (usually based on the neighborhood radius and learning rate). 
>5. Reduce the learning factor monotonically (linearly). 
>6. Reduce the radius vicinity monotonically (linearly). 
>7. Repeat steps from step 2 until the update of the weights are very few.

Dependencies
----

Python 2.7.+

Version
----

1.0

References
-----------

Below are some references:

* [Marcel Pinheiro Caraciolo] - first review on the python code (creator)
* [Python Docs] - Python Documentation
* [Wiki SOM] - Python Documentation
* [J. Gabriel Lima] - Adaptations on the code.  Any question, talk with me: **jgabriel.ufpa@gmail.com**

License
----

Apache License



[Marcel Pinheiro Caraciolo]:http://daringfireball.net/
[Python Docs]:https://docs.python.org
[Wiki SOM]:http://en.wikipedia.org/wiki/Self-organizing_map
[J. Gabriel Lima]:http://jgabriellima.com


    