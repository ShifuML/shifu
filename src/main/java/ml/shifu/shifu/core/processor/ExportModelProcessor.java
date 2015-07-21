/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.dmg.pmml.PMML;

import java.io.File;
import java.util.List;

/**
 * ExportModelProcessor class
 * 
 * @author zhanhu
 * @Nov 6, 2014
 */
public class ExportModelProcessor extends BasicModelProcessor implements Processor {

    public static final String PMML = "pmml";

    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);

    private String type;

    public ExportModelProcessor(String type) {
        this.type = type;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.processor.Processor#run()
     */
    @Override
    public int run() throws Exception {
        File pmmls = new File("pmmls");
        FileUtils.forceMkdir(pmmls);

        if(StringUtils.isBlank(type)) {
            type = PMML;
        }

        if(!type.equalsIgnoreCase(PMML)) {
            log.error("Unsupported output format - {}", type);
            return -1;
        }

        try {
            log.info("Convert models into {} format", type);

            ModelConfig modelConfig = CommonUtils.loadModelConfig();
            List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList();

            PathFinder pathFinder = new PathFinder(modelConfig);
            List<BasicML> models = CommonUtils
                    .loadBasicModels(pathFinder.getModelsPath(SourceType.LOCAL), ALGORITHM.NN);

            for(int index = 0; index < models.size(); index++) {
                log.info("\t start to generate " + "pmmls" + File.separator + modelConfig.getModelSetName()
                        + Integer.toString(index) + ".pmml");
                PMML pmml = new PMMLTranslator(modelConfig, columnConfigList, models).translate(index);
                PMMLUtils.savePMML(pmml,
                        "pmmls" + File.separator + modelConfig.getModelSetName() + Integer.toString(index) + ".pmml");
            }

            log.info("Done.");
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }

        return 0;
    }

}
