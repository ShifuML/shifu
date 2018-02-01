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
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.ModelVarSelectConf.PostCorrelationMetric;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.binning.ColumnConfigDynamicBinning;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import ml.shifu.shifu.core.binning.obj.NumericalBinInfo;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.BinaryDTSerializer;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import ml.shifu.shifu.core.dtrain.nn.BinaryNNSerializer;
import ml.shifu.shifu.core.pmml.PMMLTranslator;
import ml.shifu.shifu.core.pmml.PMMLUtils;
import ml.shifu.shifu.core.pmml.builder.PMMLConstructorFactory;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.core.varselect.ColumnStatistics;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.HDFSUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
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
    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(ExportModelProcessor.class);

    public static final String PMML = "pmml";
    public static final String COLUMN_STATS = "columnstats";
    public static final String ONE_BAGGING_MODEL = "bagging";
    public static final String ONE_BAGGING_PMML_MODEL = "baggingpmml";
    public static final String WOE_MAPPING = "woemapping";
    public static final String CORRELATION = "corr";

    public static final String IS_CONCISE = "IS_CONCISE";
    public static final String REQUEST_VARS = "REQUEST_VARS";
    public static final String EXPECTED_BIN_NUM = "EXPECTED_BIN_NUM";
    public static final String IV_KEEP_RATIO = "IV_KEEP_RATIO";
    public static final String MINIMUM_BIN_INST_CNT = "MINIMUM_BIN_INST_CNT";

    private String type;
    private Map<String, Object> params;

    /**
     * SE stats mao for correlation variable selection,if not se, this field will be null.
     */
    private Map<Integer, ColumnStatistics> seStatsMap;

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

        String modelsPath = pathFinder.getModelsPath(SourceType.LOCAL);
        if(type.equalsIgnoreCase(ONE_BAGGING_MODEL)) {
            if(!"nn".equalsIgnoreCase(modelConfig.getAlgorithm())
                    && !CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                log.warn("Currently one bagging model is only supported in NN/GBT/RF algorithm.");
            } else {
                List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                        ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));
                if(models.size() < 1) {
                    log.warn("No model is found in {}.", modelsPath);
                } else {
                    log.info("Convert nn models into one binary bagging model.");
                    Configuration conf = new Configuration();
                    Path output = new Path(pathFinder.getBaggingModelPath(SourceType.LOCAL), "model.b"
                            + modelConfig.getAlgorithm());
                    if("nn".equalsIgnoreCase(modelConfig.getAlgorithm())) {
                        BinaryNNSerializer.save(modelConfig, columnConfigList, models, FileSystem.getLocal(conf),
                                output);
                    } else if(CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                        List<List<TreeNode>> baggingTrees = new ArrayList<List<TreeNode>>();
                        for(int i = 0; i < models.size(); i++) {
                            TreeModel tm = (TreeModel) models.get(i);
                            // TreeModel only has one TreeNode instance although it is list inside
                            baggingTrees.add(tm.getIndependentTreeModel().getTrees().get(0));
                        }

                        int[] inputOutputIndex = DTrainUtils
                                .getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
                        // numerical + categorical = # of all input
                        int inputCount = inputOutputIndex[0] + inputOutputIndex[1];

                        BinaryDTSerializer.save(modelConfig, columnConfigList, baggingTrees, modelConfig.getParams()
                                .get("Loss").toString(), inputCount, FileSystem.getLocal(conf), output);
                    }
                    log.info("Please find one unified bagging model in local {}.", output);
                }
            }
        } else if(type.equalsIgnoreCase(PMML)) {
            // typical pmml generation
            List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                    ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));

            PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, isConcise(),
                    false);

            for(int index = 0; index < models.size(); index++) {
                String path = "pmmls" + File.separator + modelConfig.getModelSetName() + Integer.toString(index)
                        + ".pmml";
                log.info("\t Start to generate " + path);
                PMML pmml = translator.build(Arrays.asList(new BasicML[] { models.get(index) }));
                PMMLUtils.savePMML(pmml, path);
            }
        } else if(type.equalsIgnoreCase(ONE_BAGGING_PMML_MODEL)) {
            // one unified bagging pmml generation
            log.info("Convert models into one bagging pmml model {} format", type);
            if(!"nn".equalsIgnoreCase(modelConfig.getAlgorithm())) {
                log.warn("Currently one bagging pmml model is only supported in NN algorithm.");
            } else {
                List<BasicML> models = CommonUtils.loadBasicModels(modelsPath,
                        ALGORITHM.valueOf(modelConfig.getAlgorithm().toUpperCase()));
                PMMLTranslator translator = PMMLConstructorFactory.produce(modelConfig, columnConfigList, isConcise(),
                        true);
                String path = "pmmls" + File.separator + modelConfig.getModelSetName() + ".pmml";
                log.info("\t Start to generate one unified model to: " + path);
                PMML pmml = translator.build(models);
                PMMLUtils.savePMML(pmml, path);
            }
        } else if(type.equalsIgnoreCase(COLUMN_STATS)) {
            saveColumnStatus();
        } else if(type.equalsIgnoreCase(WOE_MAPPING)) {
            List<ColumnConfig> exportCatColumns = new ArrayList<ColumnConfig>();
            List<String> catVariables = getRequestVars();
            for(ColumnConfig columnConfig: this.columnConfigList) {
                if(CollectionUtils.isEmpty(catVariables) || isRequestColumn(catVariables, columnConfig)) {
                    exportCatColumns.add(columnConfig);
                }
            }

            if(CollectionUtils.isNotEmpty(exportCatColumns)) {
                List<String> woeMappings = new ArrayList<String>();
                for(ColumnConfig columnConfig: exportCatColumns) {
                    String woeMapText = rebinAndExportWoeMapping(columnConfig);
                    woeMappings.add(woeMapText);
                }
                FileUtils.write(new File("woemapping.txt"), StringUtils.join(woeMappings, ",\n"));
            }
        } else if (type.equalsIgnoreCase(CORRELATION)) {
            // export correlation into mapping list
            if(!ShifuFileUtils.isFileExists(pathFinder.getLocalCorrelationCsvPath(), SourceType.LOCAL)) {
                log.warn("The correlation file doesn't exist. Please make sure you have ran `shifu stats -c`.");
                return 2;
            }
            return exportVariableCorr();
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

        ColumnConfigDynamicBinning columnConfigDynamicBinning = new ColumnConfigDynamicBinning(columnConfig,
                expectBinNum, ivKeepRatio, minimumInstCnt);

        List<AbstractBinInfo> binInfos = columnConfigDynamicBinning.run();

        long[] binCountNeg = new long[binInfos.size() + 1];
        long[] binCountPos = new long[binInfos.size() + 1];
        for(int i = 0; i < binInfos.size(); i++) {
            AbstractBinInfo binInfo = binInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] = columnConfig.getBinCountNeg().get(
                columnConfig.getBinCountNeg().size() - 1);
        binCountPos[binCountPos.length - 1] = columnConfig.getBinCountPos().get(
                columnConfig.getBinCountPos().size() - 1);
        ColumnStatsCalculator.ColumnMetrics columnMetrics = ColumnStatsCalculator.calculateColumnMetrics(binCountNeg,
                binCountPos);

        System.out.println(columnConfig.getColumnName() + ":");
        for(int i = 0; i < binInfos.size(); i++) {
            if(columnConfig.isCategorical()) {
                CategoricalBinInfo binInfo = (CategoricalBinInfo) binInfos.get(i);
                System.out.println("\t" + binInfo.getValues() + " | posCount:" + binInfo.getPositiveCnt()
                        + " | negCount:" + binInfo.getNegativeCnt() + " | posRate:" + binInfo.getPositiveRate()
                        + " | woe:" + columnMetrics.getBinningWoe().get(i));
            } else {
                NumericalBinInfo binInfo = (NumericalBinInfo) binInfos.get(i);
                System.out.println("\t[" + binInfo.getLeftThreshold() + ", " + binInfo.getRightThreshold() + ")"
                        + " | posCount:" + binInfo.getPositiveCnt() + " | negCount:" + binInfo.getNegativeCnt()
                        + " | posRate:" + binInfo.getPositiveRate() + " | woe:" + columnMetrics.getBinningWoe().get(i));
            }
        }
        System.out.println("\t" + columnConfig.getColumnName() + " IV:" + columnMetrics.getIv());
        System.out.println("\t" + columnConfig.getColumnName() + " KS:" + columnMetrics.getKs());
        System.out.println("\t" + columnConfig.getColumnName() + " WOE:" + columnMetrics.getWoe());
        return generateWoeMapping(columnConfig, binInfos, columnMetrics);
    }

    private String generateWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> binInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        if(columnConfig.isCategorical()) {
            return generateCategoricalWoeMapping(columnConfig, binInfos, columnMetrics);
        } else {
            return generateNumericalWoeMapping(columnConfig, binInfos, columnMetrics);
        }
    }

    private String generateNumericalWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> numericalBinInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        buffer.append("\twhen " + columnConfig.getColumnName() + " = . then "
                + columnMetrics.getBinningWoe().get(columnMetrics.getBinningWoe().size() - 1) + "\n");
        for(int i = 0; i < numericalBinInfos.size(); i++) {
            NumericalBinInfo binInfo = (NumericalBinInfo) numericalBinInfos.get(i);
            buffer.append("\twhen (");
            if(!Double.isInfinite(binInfo.getLeftThreshold())) {
                buffer.append(binInfo.getLeftThreshold() + " <= ");
            }
            buffer.append(columnConfig.getColumnName());
            if(!Double.isInfinite(binInfo.getRightThreshold())) {
                buffer.append(" < " + binInfo.getRightThreshold());
            }
            buffer.append(") then " + columnMetrics.getBinningWoe().get(i) + "\n");
        }
        buffer.append("  end ) as " + columnConfig.getColumnName() + "_" + numericalBinInfos.size());
        return buffer.toString();
    }

    private String generateCategoricalWoeMapping(ColumnConfig columnConfig, List<AbstractBinInfo> categoricalBinInfos,
            ColumnStatsCalculator.ColumnMetrics columnMetrics) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("( case \n");
        for(int i = 0; i < categoricalBinInfos.size(); i++) {
            CategoricalBinInfo binInfo = (CategoricalBinInfo) categoricalBinInfos.get(i);
            List<String> values = new ArrayList<String>();
            for(String cval: binInfo.getValues()) {
                List<String> subCvals = CommonUtils.flattenCatValGrp(cval);
                for(String subCval: subCvals) {
                    values.add("'" + subCval + "'");
                }
            }
            buffer.append("\twhen " + columnConfig.getColumnName() + " in (" + StringUtils.join(values, ',')
                    + ") then " + columnMetrics.getBinningWoe().get(i) + "\n");
        }
        buffer.append("\telse " + columnMetrics.getBinningWoe().get(columnMetrics.getBinningWoe().size() - 1) + "\n");
        buffer.append("  end ) as " + columnConfig.getColumnName() + "_" + categoricalBinInfos.size());
        return buffer.toString();
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

    private int exportVariableCorr() throws IOException {
        Set<VarCorrInfo> varCorrInfoSet = new HashSet<VarCorrInfo>();
        BufferedReader reader = ShifuFileUtils.getReader(pathFinder.getLocalCorrelationCsvPath(), SourceType.LOCAL);
        PostCorrelationMetric metric = this.modelConfig.getVarSelect().getPostCorrelationMetric();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        try {
            int lineNum = 0;
            String line = null;
            while((line = reader.readLine()) != null) {
                lineNum += 1;
                if (lineNum <= 2) {
                    // skip first 2 lines which are indexes and names
                    continue;
                }
                String[] columns = CommonUtils.split(line, ",");
                if (columns != null && columns.length == columnConfigList.size() + 2) {
                    int columnIndex = Integer.parseInt(columns[0].trim());
                    ColumnConfig fromConfig = this.columnConfigList.get(columnIndex);
                    if (fromConfig.isTarget() || CommonUtils.isGoodCandidate(fromConfig, hasCandidates)) {
                        double[] corrArray = getCorrArray(columns);
                        for (int i = 0; i < corrArray.length; i++) {
                            ColumnConfig toConfig = this.columnConfigList.get(i);
                            if (i != columnIndex && !toConfig.isTarget() && !toConfig.isMeta()) {
                                varCorrInfoSet.add(new VarCorrInfo(fromConfig.getColumnName(),
                                        toConfig.getColumnName(), corrArray[i],
                                        getColumnMetric(fromConfig, metric), getColumnMetric(toConfig, metric)));
                            }
                        }
                    }
                }
            }
        } finally {
            IOUtils.closeQuietly(reader);
        }

        List<VarCorrInfo> varCorrInfoList = new ArrayList<VarCorrInfo>(varCorrInfoSet);
        Collections.sort(varCorrInfoList);

        String corrExportPath = this.pathFinder.getCorrExportPath();
        ShifuFileUtils.writeLines(varCorrInfoList, corrExportPath, SourceType.LOCAL);
        log.info("Done. The correlations are exported to {}", corrExportPath);

        return 0;
    }

    private double getColumnMetric(ColumnConfig config, PostCorrelationMetric metric) throws IOException {
        if ( metric == null || metric.equals(PostCorrelationMetric.IV) ) {
            // default is iv, if no PostCorrelationMetric specified
            return (config.getIv() == null ? Double.NaN : config.getIv());
        } else if ( metric.equals(PostCorrelationMetric.KS) ) {
            return (config.getKs() == null ? Double.NaN : config.getKs());
        } else if ( metric.equals(PostCorrelationMetric.SE) ) {
            if ( this.seStatsMap == null ) {
                SourceType source = this.modelConfig.getDataSet().getSource();
                String varSelectMSEOutputPath = pathFinder.getVarSelectMSEOutputPath(source);
                this.seStatsMap = readSEValuesToMap(varSelectMSEOutputPath + Path.SEPARATOR
                        + Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME + "-*", source);
            }

            return this.seStatsMap.get(config.getColumnNum()).getRms();
        }
        return -1.0d;
    }

    //TODO duplicate code with VarSelectModelProcessor. Needs do code refactor
    private Map<Integer, ColumnStatistics> readSEValuesToMap(String seOutputFiles, SourceType source)
            throws IOException {
        // here only works for 1 reducer
        FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source).globStatus(new Path(seOutputFiles));
        if(globStatus == null || globStatus.length == 0) {
            throw new RuntimeException("Var select MSE stats output file not exist.");
        }
        Map<Integer, ColumnStatistics> map = new HashMap<Integer, ColumnStatistics>();
        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            for(Scanner scanner: scanners) {
                String str = null;
                while(scanner.hasNext()) {
                    str = scanner.nextLine().trim();
                    String[] splits = CommonUtils.split(str, "\t");
                    if(splits.length == 5) {
                        map.put(Integer.parseInt(splits[0].trim()), new ColumnStatistics(Double.parseDouble(splits[2]),
                                Double.parseDouble(splits[3]), Double.parseDouble(splits[4])));
                    }
                }
            }
        } finally {
            if(scanners != null) {
                for(Scanner scanner: scanners) {
                    if(scanner != null) {
                        scanner.close();
                    }
                }
            }
        }
        return map;
    }

    private double[] getCorrArray(String[] columns) {
        double[] corr = new double[columns.length - 2];
        for(int i = 2; i < corr.length; i++) {
            corr[i - 2] = Double.parseDouble(columns[i].trim());
        }
        return corr;
    }

    private boolean isConcise() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(IS_CONCISE) instanceof Boolean) {
            return (Boolean) this.params.get(IS_CONCISE);
        }
        return false;
    }

    private List<String> getRequestVars() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(REQUEST_VARS) instanceof String) {
            String requestVars = (String) this.params.get(REQUEST_VARS);
            if(StringUtils.isNotBlank(requestVars)) {
                return Arrays.asList(requestVars.split(","));
            }
        }
        return null;
    }

    private int getExpectBinNum() {
        if(MapUtils.isNotEmpty(this.params) && this.params.get(EXPECTED_BIN_NUM) instanceof String) {
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
        if(MapUtils.isNotEmpty(this.params) && this.params.get(IV_KEEP_RATIO) instanceof String) {
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
        if(MapUtils.isNotEmpty(this.params) && this.params.get(MINIMUM_BIN_INST_CNT) instanceof String) {
            String minimumBinInstCnt = (String) this.params.get(MINIMUM_BIN_INST_CNT);
            try {
                return Long.parseLong(minimumBinInstCnt);
            } catch (Exception e) {
                log.warn("Invalid minimum bin instance count {}. Ignore it...", minimumBinInstCnt);
            }
        }
        return 0;
    }

    public static class VarCorrInfo implements Comparable<VarCorrInfo>{
        private String leftVarName;
        private String rightVarName;
        private double corrVal;
        private double leftMetricVal;
        private double rightMetricVal;

        public VarCorrInfo(String fromVarName, String toVarName, double corrVal, double fromMetricVal, double toMetricVal) {
            if ( fromVarName.compareTo(toVarName) < 0 ) {
                this.leftVarName = fromVarName;
                this.rightVarName = toVarName;
                this.leftMetricVal = fromMetricVal;
                this.rightMetricVal = toMetricVal;
            } else {
                this.leftVarName = toVarName;
                this.rightVarName = fromVarName;
                this.leftMetricVal = toMetricVal;
                this.rightMetricVal = fromMetricVal;
            }
            this.corrVal = corrVal;
        }

        @Override
        public String toString() {
            return leftVarName + "," + rightVarName + "," + corrVal + "," + leftMetricVal + "," + rightMetricVal;
        }

        @Override
        public int hashCode() {
            return this.leftVarName.hashCode() * this.rightVarName.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if ( obj == this ) {
                return true;
            }

            if (!(obj instanceof VarCorrInfo)) {
                return false;
            }

            VarCorrInfo other = (VarCorrInfo) obj;
            return this.leftVarName.equalsIgnoreCase(other.leftVarName)
                    && this.rightVarName.equalsIgnoreCase(other.rightVarName);
        }

        @Override
        public int compareTo(VarCorrInfo other) {
            // order by corrVal desc
            return Double.compare(other.corrVal, this.corrVal);
        }
    }
}
