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

import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.binning.CateDynamicBinning;
import ml.shifu.shifu.core.binning.CategoricalBinInfo;
import ml.shifu.shifu.core.binning.ColumnConfigDynamicBinning;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.core.pmml.builder.PMMLConstructorFactory;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.HDFSUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * ExportModelProcessor class
 * 
 * @author zhanhu
 */
public class ExportModelProcessor extends BasicModelProcessor implements Processor {

    public static final String PMML = "pmml";
    public static final String COLUMN_STATS = "columnstats";
    public static final String WOE_MAPPING = "woemapping";

    public static final String IS_CONCISE = "IS_CONCISE";
    public static final String REQUEST_VARS = "REQUEST_VARS";
    public static final String EXPECTED_BIN_NUM = "EXPECTED_BIN_NUM";
    public static final String IV_KEEP_RATIO = "IV_KEEP_RATIO";
    public static final String MINIMUM_BIN_INST_CNT = "MINIMUM_BIN_INST_CNT";


    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);

    private String type;
    private Map<String, Object> params;

    public ExportModelProcessor(String type, Map<String, Object> params) {
        this.type = type;
        this.params = params;
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

            PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, isConcise());

            for(int index = 0; index < models.size(); index++) {
                log.info("\t start to generate " + "pmmls" + File.separator + modelConfig.getModelSetName()
                        + Integer.toString(index) + ".pmml");
                PMML pmml = translator.build(models.get(index));
                PMMLUtils.savePMML(pmml,
                        "pmmls" + File.separator + modelConfig.getModelSetName() + Integer.toString(index) + ".pmml");
            }
        } else if(type.equalsIgnoreCase(COLUMN_STATS)) {
            saveColumnStatus();
        } else if(type.equalsIgnoreCase(WOE_MAPPING)) {
            List<ColumnConfig> exportCatColumns = new ArrayList<ColumnConfig>();
            List<String> catVariables = getRequestVars();
            for ( ColumnConfig columnConfig : this.columnConfigList ) {
                if ( columnConfig.isCategorical() ) {
                    if ( CollectionUtils.isEmpty(catVariables) || isRequestColumn(catVariables, columnConfig)) {
                        exportCatColumns.add(columnConfig);
                    }
                }
            }

            if ( CollectionUtils.isNotEmpty(exportCatColumns) ) {
                List<String> woeMappings = new ArrayList<String>();
                for ( ColumnConfig columnConfig : exportCatColumns ) {
                    String woeMapText = rebinAndExportWoeMapping(columnConfig);
                    woeMappings.add(woeMapText);
                }
                FileUtils.write(new File("woemapping.txt"), StringUtils.join(woeMappings, ",\n"));
            }
        } else {
            log.error("Unsupported output format - {}", type);
            status = -1;
        }

        clearUp(ModelStep.EXPORT);

        log.info("Done.");

        return status;
    }

    private String rebinAndExportWoeMapping(ColumnConfig columnConfig) throws IOException {
        int expectBinNum = getExpectBinNum();
        double ivKeepRatio = getIvKeepRatio();
        long minimumInstCnt = getMinimumInstCnt();

        ColumnConfigDynamicBinning columnConfigDynamicBinning =
                new ColumnConfigDynamicBinning(columnConfig, expectBinNum, ivKeepRatio, minimumInstCnt);

        List<CategoricalBinInfo> categoricalBinInfos = columnConfigDynamicBinning.run();

        long[] binCountNeg = new long[categoricalBinInfos.size() + 1];
        long[] binCountPos = new long[categoricalBinInfos.size() + 1];
        for (int i = 0; i < categoricalBinInfos.size(); i++) {
            CategoricalBinInfo binInfo = categoricalBinInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] =
                columnConfig.getBinCountNeg().get(columnConfig.getBinCountNeg().size() - 1);
        binCountPos[binCountPos.length - 1] =
                columnConfig.getBinCountPos().get(columnConfig.getBinCountPos().size() - 1);
        ColumnStatsCalculator.ColumnMetrics columnMetrics =
                ColumnStatsCalculator.calculateColumnMetrics(binCountNeg, binCountPos);

        System.out.println(columnConfig.getColumnName() + ":");
        for (int i = 0; i < categoricalBinInfos.size(); i++) {
            CategoricalBinInfo binInfo = categoricalBinInfos.get(i);
            System.out.println("\t" + binInfo.getValues()
                    + " | posCount:" + binInfo.getPositiveCnt()
                    + " | negCount:" + binInfo.getNegativeCnt()
                    + " | posRate:" + binInfo.getPositiveRate()
                    + " | woe:" + columnMetrics.getBinningWoe().get(i));
        }
        System.out.println("\t" + columnConfig.getColumnName() + " IV:" + columnMetrics.getIv());
        System.out.println("\t" + columnConfig.getColumnName() + " KS:" + columnMetrics.getKs());
        System.out.println("\t" + columnConfig.getColumnName() + " WOE:" + columnMetrics.getWoe());
        return generateWoeMapping(columnConfig.getColumnName(), categoricalBinInfos, columnMetrics, expectBinNum);
    }

    private String generateWoeMapping(String varName, List<CategoricalBinInfo> categoricalBinInfos,
                                      ColumnStatsCalculator.ColumnMetrics columnMetrics, int expectBinNum) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        for ( int i = 0; i < categoricalBinInfos.size(); i ++ ) {
            CategoricalBinInfo binInfo = categoricalBinInfos.get(i);
            List<String> values = new ArrayList<String>();
            for ( String cval : binInfo.getValues() ) {
                List<String> subCvals = CommonUtils.flattenCatValGrp(cval);
                for ( String subCval : subCvals ) {
                    values.add("'" + subCval + "'");
                }
            }
            buffer.append("\twhen " + varName + " in (" + StringUtils.join(values,',')
                    + ") then " + columnMetrics.getBinningWoe().get(i) + "\n");
        }
        buffer.append("\telse " + columnMetrics.getBinningWoe().get(columnMetrics.getBinningWoe().size() - 1) + "\n");
        buffer.append("  end ) as " + varName + "_" + categoricalBinInfos.size());
        return buffer.toString();
    }

    private String generateWoeMapping(ColumnConfig columnConfig, int expectBinNum) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        for ( int i = 0; i < columnConfig.getBinCategory().size(); i ++ ) {
            List<String> values = new ArrayList<String>();
            String cval = columnConfig.getBinCategory().get(i);
            List<String> subCvals = CommonUtils.flattenCatValGrp(cval);
            for ( String subCval : subCvals ) {
                values.add("'" + subCval + "'");
            }

            buffer.append("\twhen " + columnConfig.getColumnName() + " in (" + StringUtils.join(values,',')
                    + ") then " + columnConfig.getBinCountWoe().get(i) + "\n");
        }
        buffer.append("\telse " + columnConfig.getBinCountWoe().get(columnConfig.getBinCountWoe().size() - 1) + "\n");
        buffer.append("  end ) as " + columnConfig.getColumnName() + "_" + expectBinNum);
        return buffer.toString();
    }

    private List<CategoricalBinInfo> genCategoricalBinInfos(ColumnConfig columnConfig) {
        List<CategoricalBinInfo> categoricalBinInfos = new ArrayList<CategoricalBinInfo>();
        for ( int i = 0; i < columnConfig.getBinCategory().size(); i ++ ) {
            CategoricalBinInfo binInfo = new CategoricalBinInfo();
            List<String> values = new ArrayList<String>();
            values.add(columnConfig.getBinCategory().get(i));
            binInfo.setValues(values);
            binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(i));
            binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(i));
            binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(i));
            binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(i));

            categoricalBinInfos.add(binInfo);
        }

        // add missing binning
        CategoricalBinInfo binInfo = new CategoricalBinInfo();
        binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(columnConfig.getBinCountPos().size() - 1));
        binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(columnConfig.getBinCountNeg().size() - 1));
        binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(columnConfig.getBinWeightedPos().size() - 1));
        binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(columnConfig.getBinWeightedNeg().size() - 1));
        categoricalBinInfos.add(binInfo);

        return categoricalBinInfos;
    }

    private boolean isRequestColumn(List<String> catVariables, ColumnConfig columnConfig) {
        boolean status = false;
        for ( String varName : catVariables ) {
            if (NSColumnUtils.isColumnEqual(varName, columnConfig.getColumnName()) ) {
                status = true;
                break;
            }
        }
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

    private boolean isConcise() {
        if ( MapUtils.isNotEmpty(this.params) && this.params.get(IS_CONCISE) instanceof Boolean ) {
            return (Boolean) this.params.get(IS_CONCISE);
        }
        return false;
    }

    private List<String> getRequestVars() {
        if ( MapUtils.isNotEmpty(this.params) && this.params.get(REQUEST_VARS) instanceof String ) {
            String requestVars = (String) this.params.get(REQUEST_VARS);
            if ( StringUtils.isNotBlank(requestVars) ) {
                return Arrays.asList(requestVars.split(","));
            }
        }
        return null;
    }

    private int getExpectBinNum() {
        if ( MapUtils.isNotEmpty(this.params) && this.params.get(EXPECTED_BIN_NUM) instanceof String) {
            String expectBinNum = (String) this.params.get(EXPECTED_BIN_NUM);
            try {
                return Integer.parseInt(expectBinNum);
            } catch (Exception e) {
                log.warn("Invalid expect bin num {}. Ignore it...", expectBinNum);
            }
        }
        return 0;
    }

    public double getIvKeepRatio() {
        if ( MapUtils.isNotEmpty(this.params) && this.params.get(IV_KEEP_RATIO) instanceof String) {
            String ivKeepRatio = (String) this.params.get(IV_KEEP_RATIO);
            try {
                return Double.parseDouble(ivKeepRatio);
            } catch (Exception e) {
                log.warn("Invalid IV Keep ratio {}. Ignore it...", ivKeepRatio);
            }
        }
        return 1.0;
    }

    public long getMinimumInstCnt() {
        if ( MapUtils.isNotEmpty(this.params) && this.params.get(MINIMUM_BIN_INST_CNT) instanceof String) {
            String minimumBinInstCnt = (String) this.params.get(MINIMUM_BIN_INST_CNT);
            try {
                return Long.parseLong(minimumBinInstCnt);
            } catch (Exception e) {
                log.warn("Invalid minimum bin instance count {}. Ignore it...", minimumBinInstCnt);
            }
        }
        return 0;
    }
}
