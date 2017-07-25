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

import java.util.List;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.io.IOException;

import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.pmml.builder.impl.TreeEnsemblePmmlCreator;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Application;
import org.dmg.pmml.Header;
import org.encog.ml.BasicML;

import org.apache.pig.impl.util.JarManager;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TreeEnsemblePMMLTranslator extends PMMLTranslator {

    private static final Logger LOG = LoggerFactory.getLogger(TreeEnsemblePMMLTranslator.class);

    private TreeEnsemblePmmlCreator modelCreator;
    private AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator;
    @SuppressWarnings("unused")
    private AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator;

    public TreeEnsemblePMMLTranslator(TreeEnsemblePmmlCreator modelCreator,
            AbstractPmmlElementCreator<DataDictionary> dataDictionaryCreator,
            AbstractPmmlElementCreator<MiningSchema> miningSchemaCreator) {
        super();
        this.modelCreator = modelCreator;
        this.dataDictionaryCreator = dataDictionaryCreator;
        this.miningSchemaCreator = miningSchemaCreator;
    }

    public PMML build(BasicML basicML) {
        PMML pmml = new PMML();

        Header header = new Header();
        pmml.setHeader(header);
        header.setCopyright(" Copyright [2013-2017] PayPal Software Foundation\n" + "\n"
                + " Licensed under the Apache License, Version 2.0 (the \"License\");\n"
                + " you may not use this file except in compliance with the License.\n"
                + " You may obtain a copy of the License at\n" + "\n"
                + "    http://www.apache.org/licenses/LICENSE-2.0\n" + "\n"
                + " Unless required by applicable law or agreed to in writing, software\n"
                + " distributed under the License is distributed on an \"AS IS\" BASIS,\n"
                + " WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
                + " See the License for the specific language governing permissions and\n"
                + " limitations under the License.\n");
        Application application = new Application();
        header.setApplication(application);

        String findContainingJar = JarManager.findContainingJar(TreeEnsemblePMMLTranslator.class);
        JarFile jar = null;
        try {
            jar = new JarFile(findContainingJar);
            final Manifest manifest = jar.getManifest();
            String vendor = manifest.getMainAttributes().getValue("vendor");
            application.setName(vendor);
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
        pmml.setDataDictionary(dataDictionaryCreator.build(basicML));
        List<Model> models = pmml.getModels();
        Model miningModel = modelCreator.convert(((TreeModel) basicML).getIndependentTreeModel());
        models.add(miningModel);
        return pmml;
    }
}
