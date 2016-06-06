package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.core.LR;
import ml.shifu.shifu.core.pmml.PMMLLRModelBuilder;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import org.dmg.pmml.Model;
import org.dmg.pmml.RegressionModel;
import org.encog.ml.BasicML;

/**
 * Created by zhanhu on 3/29/16.
 */
public class RegressionSpecifCreator extends AbstractSpecifCreator {

    @Override
    public boolean build(BasicML basicML, Model model) {
        RegressionModel regression = (RegressionModel)model;
        new PMMLLRModelBuilder().adaptMLModelToPMML((LR)basicML, regression);
        regression.withOutput(createNormalizedOutput());
        return true;
    }

}
