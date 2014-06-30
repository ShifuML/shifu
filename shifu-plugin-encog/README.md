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

1.Adapt the [Encog BasicNetwork model](https://github.com/lisahua/shifu/blob/develop/shifu-plugin-encog/src/test/java/ml/shifu/plugin/encog/adapter/PMMLEncogNeuralNetworkTest.java) to PMML model, get the PMML model object

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
        String path = "src/test/resources/spark/lr/SparkLR.pmml";
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
2.Specific Model Convertor

```
public class PMMLEncogNeuralNetworkModel implements  PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, BasicNetwork> {

    public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(org.encog.neural.networks.BasicNetwork bNetwork,
            org.dmg.pmml.NeuralNetwork pmmlModel) {
       ...
```



## Extend PMMLAdapter

1.  Extend  ``` pmmlAdapter.PMMLModelGenerator``` for this new ML framework. ```PMMLModelGenerator``` contains APIs for all PMML models. 

* Override the APIs in the ```PMMLModelGenerator``` for the corresponding PMML models you prefer to convert to. You can convert a machine learning model to a different type of PMML model. For instance, in Encog, Logistic Regression model is regarded as a special case of Neural Network. You can adapt a neural network from Encog, which has only input layer and output layer, to a Logistic Regression model in PMML.
2. Extend ```PMMLModelBuilder<TargetPMMLModel, SourceMLModel>``` 

 2.1 Initialize the ```TargetPMMLModel pmmlModel``` in the specific ```PMMLModelBuilder```   
 
 2.2 Extend the function ```protected  void adaptMLModelToPMML(SourceMLModel source);``` to finish the adaptation

Here is an example of ```EncogPMMLModelGenerator```. 

### Create a PMMLModelGenerator for the ML framework

Suppose you want to convert Encog neural network model to PMML logistic regression model, you can create a ```EncogPMMLModelGenerator``` for the Encog ML framework.
```
public class EncogPMMLModelGenerator extends PMMLModelGenerator {
@Override
    public RegressionModel createRegressionModel(Object regressionModel) {
        return new PMMLEncogLogisticRegressionModel()
                .convertMLModelToPMML((BasicNetwork) regressionModel,utility);
    }
    ...
}
```
#### Implmenet PMMLModelBuilder<TargetPMMLModel, SourceMLModel>

You create a specific PMMLModelBuilder ```PMMLEncogLogisticRegressionModel``` which extends from ``` PMMLModelBuilder<pmml.RegressionModel, encog.BasicNetwork>```. 

1. Initialize the object ```pmmlModel``` as the PMML model you prefer to get.
2. Implement the adaptation method ```adaptMLModelToPMML(BasicNetwork bNetwork)``` which does the adaptation work.

```
public class PMMLEncogLogisticRegressionModel extends 
        PMMLModelBuilder<RegressionModel, BasicNetwork> {
    private FlatNetwork network;
    public PMMLEncogLogisticRegressionModel() {
    //initialize PMML model
        pmmlModel = new RegressionModel();
    }
    //implement the adaptation function
    protected void adaptMLModelToPMML(BasicNetwork bNetwork) {
        network = bNetwork.getFlat();
        double[] weights = network.getWeights();
        RegressionTable table = new RegressionTable()  ;
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION)
                .withNormalizationMethod(
                        RegressionNormalizationMethodType.LOGIT);
       ...
        table.withNumericPredictors(new NumericPredictor(new FieldName(utility.getBIAS()),
                weights[index++]));
        pmmlModel.withRegressionTables(table);
    }
}
```
* Notice that, the PMMLModelBuilder may require some supplementary data fields, which may not included in the Source ML model (eg, Column names). These information is stored in ```DataFieldRequestCoordinator``` that can be extended to meet different requirements for different ML frameworks.

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







