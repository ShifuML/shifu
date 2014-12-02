/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.processor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelOutput;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperMasterConductor;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperWorkerConductor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.data.MLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * Variable selection processor, select the variable based on KS/IV value, or </p>
 * <p/>
 * Selection variable based on the wrapper training processor.
 * </p>
 */
public class VarSelectModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    public static final String SHIFU_DEFAULT_DTRAIN_PARALLEL = "true";

    /**
     * run for the variable selection
     */
    @Override
    public int run() throws Exception {
        setUp(ModelStep.VARSELECT);

        if(modelConfig.getVotedVariablesSelection()) {
            votedVariablesSelection();
        } else {
            nativeVarialeSelection();
        }

        clearUp(ModelStep.VARSELECT);
        return 0;
    }

    private int nativeVarialeSelection() throws Exception {

        CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

        VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);

        // Filter
        this.columnConfigList = selector.selectByFilter();
        try {
            saveColumnConfigList();
        } catch (ShifuException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
        }

        // Wrapper, only if enabled
        if(modelConfig.getVarSelectWrapperEnabled()) {
            wrapper(selector);
        }
        log.info("Step Finished: varselect");

        return 0;
    }

    private void votedVariablesSelection() throws ClassNotFoundException, IOException, InterruptedException {
        log.info("Start voted variables selection ");
        //sync data back to hdfs
        super.syncDataToHdfs(modelConfig.getDataSet().getSource());
        
        SourceType sourceType = super.getModelConfig().getDataSet().getSource();

        final List<String> args = new ArrayList<String>();
        // prepare parameter
        prepareVarSelParams(args, sourceType);

        Path columnIdsPath = getVotedSelectionPath(sourceType);
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, Constants.VAR_SEL_COLUMN_IDS_OUPUT,
                columnIdsPath.toString()));

        long start = System.currentTimeMillis();

        GuaguaMapReduceClient guaguaClient = new GuaguaMapReduceClient();

        guaguaClient.creatJob(args.toArray(new String[0])).waitForCompletion(true);

        log.info("Voted variables selection finished in {}ms.", System.currentTimeMillis() - start);

        persistColumnIds(columnIdsPath);
        super.syncDataToHdfs(sourceType);
    }

    private int persistColumnIds(Path path) {
        try {
            List<Scanner> scanners = ShifuFileUtils.getDataScanners(path.toString(), modelConfig.getDataSet()
                    .getSource());

            List<Integer> ids = null;
            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String[] raw = scanner.nextLine().trim().split("\\|");

                    int idSize = Integer.valueOf(raw[0]);

                    ids = CommonUtils.stringToIntegerList(raw[1]);

                }
            }

            // prevent multiply running setting
            for(ColumnConfig config: columnConfigList) {
                if(!config.isForceSelect()) {
                    config.setFinalSelect(Boolean.FALSE);
                }
            }

            for(Integer id: ids) {
                this.columnConfigList.get(id).setFinalSelect(Boolean.TRUE);
            }

            super.saveColumnConfigList();

        } catch (IOException e) {
            e.printStackTrace();
            return -1;
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            return -1;
        }

        return 0;
    }

    private Path getVotedSelectionPath(SourceType sourceType) {

        return ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels"));
    }

    private void prepareVarSelParams(final List<String> args, final SourceType sourceType) {
        args.add("-libjars");

        addRuntimeJars(args);

        args.add("-i");
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(modelConfig.getDataSetRawPath())).toString());

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            log.warn("No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embeded zookeeper server in client process. For big data applications, specified zookeeper servers are strongly recommended.");
        } else {
            args.add("-z");
            args.add(zkServers);
        }

        // setting the class
        args.add("-w");
        args.add(VarSelWorker.class.getName());

        args.add("-m");
        args.add(VarSelMaster.class.getName());

        args.add("-c");
        // the reason to add 1 is that the first iteration in D-NN implementation is used for training preparation.
        // FIXME, how to set iteration number
        int expectVarCount = this.modelConfig.getVarSelectFilterNum();
        int forceSelectCount = 0;
        int candidateCount = 0;
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isForceSelect()) {
                forceSelectCount++;
            }
            if(CommonUtils.isGoodCandidate(columnConfig)) {
                candidateCount++;
            }
        }

        args.add(String.valueOf(Math.min(expectVarCount, candidateCount) - forceSelectCount + 1));

        args.add("-mr");
        args.add(VarSelMasterResult.class.getName());

        args.add("-wr");
        args.add(VarSelWorkerResult.class.getName());

        // setting conductor
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, Constants.VAR_SEL_MASTER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperMasterConductor.class.getName())));

        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, Constants.VAR_SEL_WORKER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperWorkerConductor.class.getName())));

        // setting queue
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE)));

        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                VarSelOutput.class.getName()));

        // setting model config column config
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));

        // source type
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_MODELSET_SOURCE_TYPE, sourceType));

        // computation time
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD,
                60 * 60 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(entry.getKey().toString().startsWith("nn") || entry.getKey().toString().startsWith("guagua")
                    || entry.getKey().toString().startsWith("mapred")) {
                args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(), entry.getValue()
                        .toString()));
            }
        }
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
    private void addRuntimeJars(final List<String> args) {
        List<String> jars = new ArrayList<String>(16);
        // jackson-databind-*.jar
        jars.add(JarManager.findContainingJar(ObjectMapper.class));
        // jackson-core-*.jar
        jars.add(JarManager.findContainingJar(JsonParser.class));
        // jackson-annotations-*.jar
        jars.add(JarManager.findContainingJar(JsonIgnore.class));
        // commons-compress-*.jar
        jars.add(JarManager.findContainingJar(BZip2CompressorInputStream.class));
        // commons-lang-*.jar
        jars.add(JarManager.findContainingJar(StringUtils.class));
        // commons-collections-*.jar
        jars.add(JarManager.findContainingJar(ListUtils.class));
        // common-io-*.jar
        jars.add(JarManager.findContainingJar(org.apache.commons.io.IOUtils.class));
        // guava-*.jar
        jars.add(JarManager.findContainingJar(Splitter.class));
        // encog-core-*.jar
        jars.add(JarManager.findContainingJar(MLDataSet.class));
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // guagua-core-*.jar
        jars.add(JarManager.findContainingJar(GuaguaConstants.class));
        // guagua-mapreduce-*.jar
        jars.add(JarManager.findContainingJar(GuaguaMapReduceConstants.class));
        // zookeeper-*.jar
        jars.add(JarManager.findContainingJar(ZooKeeper.class));

        jars.add(JarManager.findContainingJar(JexlException.class));

        args.add(StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR));
    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
        // "-Xmn128m -Xms1G -Xmx1G -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                "-Xmn128m -Xms1G -Xmx1G"));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, SHIFU_DEFAULT_DTRAIN_PARALLEL)));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456")));
    }

    /**
     * user wrapper to select variable
     * 
     * @param selector
     * @throws Exception
     */
    private void wrapper(VariableSelector selector) throws Exception {

        NormalizeModelProcessor n = new NormalizeModelProcessor();

        // runNormalize();
        n.run();

        TrainModelProcessor t = new TrainModelProcessor(false, false);
        t.run();

        AbstractTrainer trainer = t.getTrainer(0);

        if(trainer instanceof NNTrainer) {
            selector.selectByWrapper((NNTrainer) trainer);
            try {
                this.saveColumnConfigList();
            } catch (ShifuException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
            }
        }
    }

}
