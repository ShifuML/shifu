package ml.shifu.core.di.builtin;

import ml.shifu.core.di.spi.PMMLModelElementCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;
import org.dmg.pmml.NeuralNetwork;

public class DefaultPMMLModelElementCreator implements PMMLModelElementCreator {

    public Model create(Params params) {

        String modelType = params.get("modelType").toString();
        String modelName = params.get("modelName").toString();

        Model model = null;

        if (modelType.equalsIgnoreCase("NeuralNetwork")) {
            model = new NeuralNetwork();
        } else {
            throw new RuntimeException("Model not supported: " + modelType);
        }

        model.setModelName(modelName);

        return model;
    }

}
