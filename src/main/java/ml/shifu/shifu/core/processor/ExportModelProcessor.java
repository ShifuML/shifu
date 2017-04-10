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

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.core.pmml.builder.PMMLConstructorFactory;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ExportModelProcessor class
 * 
 * @author zhanhu
 */
public class ExportModelProcessor extends BasicModelProcessor implements Processor {

    public static final String PMML = "pmml";
    public static final String COLUMN_STATS = "columnstats";

    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);

    private String type;
    private boolean isConcise;

    public ExportModelProcessor(String type, boolean isConcise) {
        this.type = type;
        this.isConcise = isConcise;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.processor.Processor#run()
     */
    @Override
    public int run() throws Exception {
        setUp(ModelStep.EXPORT);

        int status = 0;
        File pmmls = new File("pmmls");
        FileUtils.forceMkdir(pmmls);

        if(StringUtils.isBlank(type)) {
            type = PMML;
        }

        if(type.equalsIgnoreCase(PMML)) {
            log.info("Convert models into {} format", type);

            List<BasicML> models = CommonUtils.loadBasicModels(pathFinder.getModelsPath(SourceType.LOCAL),
                    ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));

            PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, this.isConcise);

            for(int index = 0; index < models.size(); index++) {
                log.info("\t start to generate " + "pmmls" + File.separator + modelConfig.getModelSetName()
                        + Integer.toString(index) + ".pmml");
                PMML pmml = translator.build(models.get(index));
                PMMLUtils.savePMML(pmml,
                        "pmmls" + File.separator + modelConfig.getModelSetName() + Integer.toString(index) + ".pmml");
            }
        } else if(type.equalsIgnoreCase(COLUMN_STATS)) {
            saveColumnStatus();
        } else {
            log.error("Unsupported output format - {}", type);
            status = -1;
        }

        clearUp(ModelStep.EXPORT);

        log.info("Done.");

        return status;
    }

    private void saveColumnStatus() throws IOException {
        Path localColumnStatsPath = new Path(pathFinder.getLocalColumnStatsPath());
        log.info("Saving ColumnStatus to local file system: {}.", localColumnStatsPath);
        if(HDFSUtils.getLocalFS().exists(localColumnStatsPath)) {
            HDFSUtils.getLocalFS().delete(localColumnStatsPath, true);
        }

        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(localColumnStatsPath.toString(), SourceType.LOCAL);
            writer.write("dataSet,columnFlag,columnName,columnNum,iv,ks,max,mean,median,min,missingCount,"
                    + "missingPercentage,stdDev,totalCount,distinctCount,weightedIv,weightedKs,weightedWoe,woe,"
                    + "skewness,kurtosis,columnType,finalSelect,psi,unitstats,version\n");
            StringBuilder builder = new StringBuilder(500);
            for(ColumnConfig columnConfig: columnConfigList) {
                builder.setLength(0);
                builder.append(modelConfig.getBasic().getName()).append(',');
                builder.append(columnConfig.getColumnFlag()).append(',');
                builder.append(columnConfig.getColumnName()).append(',');
                builder.append(columnConfig.getColumnNum()).append(',');
                builder.append(columnConfig.getIv()).append(',');
                builder.append(columnConfig.getKs()).append(',');
                builder.append(columnConfig.getColumnStats().getMax()).append(',');
                builder.append(columnConfig.getColumnStats().getMean()).append(',');
                builder.append(columnConfig.getColumnStats().getMedian()).append(',');
                builder.append(columnConfig.getColumnStats().getMin()).append(',');
                builder.append(columnConfig.getColumnStats().getMissingCount()).append(',');
                builder.append(columnConfig.getColumnStats().getMissingPercentage()).append(',');
                builder.append(columnConfig.getColumnStats().getStdDev()).append(',');
                builder.append(columnConfig.getColumnStats().getTotalCount()).append(',');
                builder.append(columnConfig.getColumnStats().getDistinctCount()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedIv()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedKs()).append(',');
                builder.append(columnConfig.getColumnStats().getWeightedWoe()).append(',');
                builder.append(columnConfig.getColumnStats().getWoe()).append(',');
                builder.append(columnConfig.getColumnStats().getSkewness()).append(',');
                builder.append(columnConfig.getColumnStats().getKurtosis()).append(',');
                builder.append(columnConfig.getColumnType()).append(',');
                builder.append(columnConfig.isFinalSelect()).append(',');
                builder.append(columnConfig.getPSI()).append(',');
                builder.append(StringUtils.join(columnConfig.getUnitStats(), '|')).append(',');
                builder.append(modelConfig.getBasic().getVersion()).append("\n");
                writer.write(builder.toString());
            }
        } finally {
            writer.close();
        }
    }
}
