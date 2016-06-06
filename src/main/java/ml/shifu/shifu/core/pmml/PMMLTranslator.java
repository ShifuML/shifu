/**
 * Copyright [2012-2015] PayPal Software Foundation
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;

public class PMMLTranslator {

    private AbstractPmmlElementCreator<Model> modelCreator;
    private AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator;
    private AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator;
    private AbstractPmmlElementCreator<ModelStats> modelStatsCreator;
    private AbstractPmmlElementCreator<LocalTransformations> localTransformationsCreator;
    private AbstractSpecifCreator specifCreator;

    public PMMLTranslator(AbstractPmmlElementCreator<Model> modelCreator,
                          AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator,
                          AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator,
                          AbstractPmmlElementCreator<ModelStats> modelStatsCreator,
                          AbstractPmmlElementCreator<LocalTransformations> localTransformationsCreator,
                          AbstractSpecifCreator specifCreator) {
        this.modelCreator = modelCreator;
        this.dataDictionaryCreator = dataDictionaryCreator;
        this.miningSchemaCreator = miningSchemaCreator;
        this.modelStatsCreator = modelStatsCreator;
        this.localTransformationsCreator = localTransformationsCreator;
        this.specifCreator = specifCreator;
    }

    public PMML build(BasicML basicML) {
        PMML pmml = new PMML();

        // create and set data dictionary
        pmml.setDataDictionary(this.dataDictionaryCreator.build());

        // create model element
        Model model = this.modelCreator.build();

        // create mining schema
        model.setMiningSchema(this.miningSchemaCreator.build());

        // create variable statistical info
        model.setModelStats(this.modelStatsCreator.build());

        // create variable transform
        model.setLocalTransformations(this.localTransformationsCreator.build());

        this.specifCreator.build(basicML, model);

        pmml.withModels(model);

        return pmml;
    }

}
