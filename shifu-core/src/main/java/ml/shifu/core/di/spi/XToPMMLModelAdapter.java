package ml.shifu.core.di.spi;


import org.dmg.pmml.Model;

public interface XToPMMLModelAdapter {

    public Model exec(Object originModel, Model partialPMMLModel);

}
