/*
 * Copyright [2013-2016] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.pmml.builder.impl;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Target;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.Targets;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.encog.ml.BasicML;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;

/**
 * Created by zhanhu on 3/29/16.
 */
public class NNPmmlModelCreator extends AbstractPmmlElementCreator<Model> {

    public NNPmmlModelCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList, false);
    }

    public NNPmmlModelCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    public Model build(BasicML basicML) {
        Model model = new NeuralNetwork();
/*        if ( modelConfig.isClassification() &&
                ModelTrainConf.MultipleClassification.NATIVE.equals(modelConfig.getTrain().getMultiClassifyMethod())) {
            model.setFunctionName(MiningFunctionType.CLASSIFICATION);
        } else {*/
            model.setMiningFunction(MiningFunction.REGRESSION);
/*        }*/
        model.setTargets(createTargets());
        return model;
    }

    public Targets createTargets() {
        Targets targets = new Targets();

        if ( modelConfig.isClassification() &&
                ModelTrainConf.MultipleClassification.NATIVE.equals(modelConfig.getTrain().getMultiClassifyMethod()) ) {
            List<Target> targetList = createMultiClassTargets();
            targets.addTargets(targetList.toArray(new Target[targetList.size()]));
        } else {
            Target target = new Target();

        target.setOpType(OpType.CATEGORICAL);
        target.setField(new FieldName(modelConfig.getTargetColumnName()));

        List<TargetValue> targetValueList = new ArrayList<TargetValue>();

        for(String posTagValue: modelConfig.getPosTags()) {
            TargetValue pos = new TargetValue();
            pos.setValue(posTagValue);
            pos.setDisplayValue("Positive");

            targetValueList.add(pos);
        }

        for(String negTagValue: modelConfig.getNegTags()) {
            TargetValue neg = new TargetValue();
            neg.setValue(negTagValue);
            neg.setDisplayValue("Negative");

            targetValueList.add(neg);
        }

        target.addTargetValues(targetValueList.toArray(new TargetValue[targetValueList.size()]));

            targets.addTargets(target);
        }

        return targets;
    }

    private List<Target> createMultiClassTargets() {
        List<Target> targets = new ArrayList<Target>();
        for ( int i = 0; i < modelConfig.getTags().size(); i ++ ) {
            String tag = modelConfig.getTags().get(i);

            Target target = new Target();
            target.setOpType(OpType.CONTINUOUS);
            target.setField(new FieldName(modelConfig.getTargetColumnName() + "_" + i));

            TargetValue targetValue = new TargetValue();
            targetValue.setValue(Integer.toString(i));
            targetValue.setDisplayValue(tag);

            target.addTargetValues(targetValue);

            targets.add(target);
        }
        return targets;
    }
}
