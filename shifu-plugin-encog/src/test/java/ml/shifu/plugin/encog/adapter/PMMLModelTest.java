package ml.shifu.plugin.encog.adapter;

import org.dmg.pmml.PMML;


public abstract class PMMLModelTest<T> extends BasicAdapterTest{
    protected final double DELTA = Math.pow(10, -5);
    protected PMML pmml;

    /**
     * Initialize the Machine Learning model in ML framework. This function must
     * be called before the rest test functions.
     */
    abstract protected void initMLModel();

    /**
     * Convert the ML model to PMML model by invoking the PMMLAdapter. This
     * function must be called after ML model is initialized in initMLModel()
     * function.
     */
    abstract protected void adaptToPMML();

    /**
     * Write the PMML model to PMML file
     */
    abstract protected void writeToPMML();

    /**
     * Compare the predict/compute result from ML model with the evaluation
     * result from PMML evaluator. This function is used to vaidate the
     * correctness of the generated PMML file.
     */
    abstract protected void evaluatePMML();

    /**
     * This function specifies the basic test flow for the test cases. That is,
     * the ML model initialization function must be called before adaptation
     * function.
     */
    public void testSetUp() {
        initMLModel();
        adaptToPMML();
    }

}
