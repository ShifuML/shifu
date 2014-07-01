package ml.shifu.plugin.encog.adapter;

import ml.shifu.core.di.spi.XToPMMLModelAdapter;
import org.dmg.pmml.Model;
import org.dmg.pmml.NeuralNetwork;
import org.encog.neural.networks.BasicNetwork;

public class EncogNeuralNetworkToPMMLAdapter implements XToPMMLModelAdapter {

    public Model exec(Object originModel, Model partialPMMLModel) {

        if (originModel instanceof BasicNetwork && partialPMMLModel instanceof NeuralNetwork) {
            return new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML(
                    (BasicNetwork) originModel, (NeuralNetwork) partialPMMLModel);
        } else {
            throw new RuntimeException("Cannot convert model.");
        }


    }


}
