## PMMLAdapter

* convert multiple Machine Learning Framework to PMML format.

| ML Framework | Neural Network | Logistic Regression |Support Vector Machine |  Decision Tree|
|-------|--------|---------|-------|---------|
|Encog| support| support| support| None|
|Spark| None| support| TBD| TBD|
|Mahout|Support| Support|TBD|TBD|
|H2o|None| TBD| TBD| TBD|

* None: the framework does not support the corresponding ML model
* TBD: the framework has the corresponding ML model, while PMMLAdapter hasn't supported the conversion of this algorithm to PMML model.

## Get Started
Here is [an example](https://github.com/lisahua/shifu/blob/develop/shifu-plugin-encog/src/test/java/ml/shifu/plugin/encog/adapter/PMMLEncogNeuralNetworkTest.java)  of how to use the [PMML Adapter](https://github.com/lisahua/shifu/blob/develop/shifu-plugin-encog/src/main/java/ml/shifu/plugin/encog/adapter/PMMLEncogNeuralNetworkModel.java).

1.Adapt the Encog BasicNetwork model to PMML model and return the PMML model object

```
    protected void adaptToPMML() {
        NeuralNetwork pmmlNN = (NeuralNetwork) pmml.getModels().get(0);
        pmmlNN = new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML(mlModel,
                pmmlNN);
        pmml.getModels().set(0, pmmlNN);
    }
```

2.Write PMML model to the file

```
    @Override
    protected void writeToPMML() {
        String path = "src/test/resources/encog/nn/EncogNN_ouptput.pmml";
        try {
            // write PMML
            OutputStream os = new FileOutputStream(path);
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


```

## Implementation 
1.Model Convertor Interface

```
/**
 * The abstract class that converts the Machine Learing model to a PMML model
 * 
 * @param <T>            The target PMML model type
 * @param <S>           The source ML model from specific Machine Learning framework such
 *            as Encog, Machout, and Spark.
 */
public interface PMMLModelBuilder<T extends Model,S> {

   T adaptMLModelToPMML(S mlModel, T partialPMMLModel);

}
```
2.Specific Model Adapter

```
public class PMMLEncogNeuralNetworkModel implements  PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, BasicNetwork> {

    public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(org.encog.neural.networks.BasicNetwork bNetwork,
            org.dmg.pmml.NeuralNetwork pmmlModel) {
       ...
```


## Extend PMMLAdapter

1. Implement a specific PMMLModelBuilder that implements PMMLModelBuilder<TargetPMMLModel,SourceMLModel> interface.
  
2. Implement the adaptation method ```adaptMLModelToPMML(SourceMLModel,PartialPMMLModel)``` that does the adaptation work.

```
public class PMMLEncogNeuralNetworkModel  implements  PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, org.encog.neural.networks.BasicNetwork> {

 public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(org.encog.neural.networks.BasicNetwork bNetwork,  org.dmg.pmml.NeuralNetwork pmmlModel) {
             ...
            return pmmlModel;
            }
```


## Test your PMMLAdapter

1. Store the input data and evaluation data in corresponding folders in ```test/resources/```
2. Extends from ```PMMLModelTest<SourceMLModel>```
3. Implment the functions below:

 3.1 ```initMLModel()``` that generates the source machine learning model.
 
 3.2  ```adaptToPMML()``` that invokes the PMMLAdapter to convert the ML model to PMML model. To make sure the ML model is initiazed before being adapted to PMML model, there is a ```testSetUp()``` function in the parent test class ```PMMLModelTest``` that defines the basic testing work flow.
 
 3.3  optional: ```writeToPMML()``` that writes the PMMLModel to the file.
 
 3.4  optional: ```evaluatePMML()``` that invokes ```PMMLEvaluator``` to compare the score calcaulated via the PMML file against the evalution score from the ML framework.

4. Create specific test functions which invokes the functions you want to test. 
```
    @Test
    public void testEncogLR() {
        testSetUp();
        writeToPMML();
    }
```







