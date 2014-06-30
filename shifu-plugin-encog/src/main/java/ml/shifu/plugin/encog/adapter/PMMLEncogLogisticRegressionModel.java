package ml.shifu.plugin.encog.adapter;

import ml.shifu.plugin.PMMLAdapterCommonUtil;
import ml.shifu.plugin.PMMLModelBuilder;

import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;

/**
 * The class that converts a special Encog NeuralNetwork without hidden layers
 * to a PMML RegressionModel. This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,Encog.NeuralNetwork>.
 * 
 */
public class PMMLEncogLogisticRegressionModel implements
        PMMLModelBuilder<org.dmg.pmml.RegressionModel, BasicNetwork> {
    private FlatNetwork network;

    /**
     * The function which converts a special Encog NeuralNetwork without hidden
     * layers, to a PMML RegressionModel.
     * 
     * @param bNetwork
     *            Encog NeuralNetwork
     * @param utility
     *            DataFieldUtility that provides supplementary data field for
     *            the model conversion
     * @return The generated PMML RegressionModel
     */
    public org.dmg.pmml.RegressionModel adaptMLModelToPMML(
            BasicNetwork bNetwork, org.dmg.pmml.RegressionModel pmmlModel) {
        network = bNetwork.getFlat();
        double[] weights = network.getWeights();
        return PMMLAdapterCommonUtil.getRegressionTable(weights, 0, pmmlModel);
    }

}
