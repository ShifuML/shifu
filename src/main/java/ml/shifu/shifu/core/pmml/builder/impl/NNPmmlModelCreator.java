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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;

import java.util.ArrayList;
import java.util.List;

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
        model.setTargets(createTargets());
        return model;
    }

    public Targets createTargets() {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOptype(OpType.CATEGORICAL);
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

        target.withTargetValues(targetValueList);

        targets.withTargets(target);

        return targets;
    }
}
