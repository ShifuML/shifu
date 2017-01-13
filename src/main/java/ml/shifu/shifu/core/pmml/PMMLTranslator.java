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
package ml.shifu.shifu.core.pmml;

import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.io.IOException;

import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import org.dmg.pmml.Application;
import org.dmg.pmml.Header;
import org.encog.ml.BasicML;

import org.apache.pig.impl.util.JarManager;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PMMLTranslator {

    private static final Logger LOG = LoggerFactory.getLogger(PMMLTranslator.class);

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

    public PMMLTranslator() {
    }

    public PMML build(BasicML basicML) {
        PMML pmml = new PMML();
        //create and add header
        Header header = new Header();
        pmml.setHeader(header);
        header.setCopyright(
        " Copyright [2013-2016] PayPal Software Foundation\n" +
        "\n" +
        " Licensed under the Apache License, Version 2.0 (the \"License\");\n" +
        " you may not use this file except in compliance with the License.\n" +
        " You may obtain a copy of the License at\n" +
        "\n" +
        "    http://www.apache.org/licenses/LICENSE-2.0\n" +
        "\n" +
        " Unless required by applicable law or agreed to in writing, software\n" +
        " distributed under the License is distributed on an \"AS IS\" BASIS,\n" +
        " WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" +
        " See the License for the specific language governing permissions and\n" +
        " limitations under the License.\n");
        Application application = new Application();
        header.setApplication(application);

        application.setName("shifu");
        String findContainingJar = JarManager.findContainingJar(TreeEnsemblePMMLTranslator.class);
        JarFile jar = null;
        try {
            jar = new JarFile(findContainingJar);
            final Manifest manifest = jar.getManifest();

            String vendor = manifest.getMainAttributes().getValue("vendor");
            String version = manifest.getMainAttributes().getValue("version");
            application.setVersion(version);
        } catch (Exception e) {
            LOG.warn(e.getMessage());
        } finally {
            if(jar != null) {
                try {
                    jar.close();
                } catch (IOException e) {
                    LOG.warn(e.getMessage());
                }
            }
        }
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
