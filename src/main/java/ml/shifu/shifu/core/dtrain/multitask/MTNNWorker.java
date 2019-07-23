package ml.shifu.shifu.core.dtrain.multitask;

import com.google.common.base.Splitter;
import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.wdl.WDLWorker;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.*;

/**
 * @author haillu
 * @date 7/17/2019 5:05 PM
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class MTNNWorker extends
        AbstractWorkerComputable<MTNNParams, MTNNParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTNNWorker.class);

    private ModelConfig modelConfig;

    private List<ColumnConfig> columnConfigList;

    protected int inputCount;

    protected int numInputs;

    protected int cateInputs;

    private boolean isAfterVarSelect = true;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    protected long sampleCount;

    private volatile MemoryLimitedList<WDLWorker.Data> trainingData;

    private volatile MemoryLimitedList<WDLWorker.Data> validationData;

    private Map<Integer, Map<String, Integer>> columnCategoryIndexMapping;

    private Splitter splitter;

    private ConcurrentMap<Integer, Integer> inputIndexMap = new ConcurrentHashMap<Integer, Integer>();


    private MultiTaskNN mtnn;

    private int trainerId = 0;

    private int workerThreadCount;

    CompletionService<MTNNParams> completionService;

    private boolean hasCandidates;

    protected boolean isManualValidation = false;

    /**
     * If stratified sampling or random sampling
     */
    private boolean isStratifiedSampling = false;

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;


    protected boolean poissonSampler;

    /**
     * PoissonDistribution which is used for poisson sampling for bagging with replacement.
     */
    protected PoissonDistribution rng = null;

    /**
     * PoissonDistribution which is used for up sampling positive records.
     */
    protected PoissonDistribution upSampleRng = null;

    /**
     * Parameters defined in ModelConfig.json#train part
     */
    private Map<String, Object> validParams;


    // todo:for eval.
    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        // initialize Hadoop based line (long, string) reader
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<MTNNParams, MTNNParams> context) {
        Properties props = context.getProps();
        try {
            RawSourceData.SourceType sourceType = RawSourceData.SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, RawSourceData.SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        String delimiter = context.getProps().getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER);
        this.splitter = MapReduceUtils.generateShifuOutputSplitter(delimiter);

        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if (kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            LOG.info("Cross validation is enabled by kCrossValidation: {}.", kCrossValidation);
        }

        this.workerThreadCount = modelConfig.getTrain().getWorkerThreadCount();
        this.completionService = new ExecutorCompletionService<>(Executors.newFixedThreadPool(workerThreadCount));

        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if (upSampleWeight != 1d && (modelConfig.isRegression()
                || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))) {
            // set mean to upSampleWeight -1 and get sample + 1to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }

        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);

        double validationRate = this.modelConfig.getValidSetRate();
        if (StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
            // fixed 0.6 and 0.4 of max memory for trainingData and validationData
            this.trainingData = new MemoryLimitedList<WDLWorker.Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.6), new ArrayList<WDLWorker.Data>());
            this.validationData = new MemoryLimitedList<WDLWorker.Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.4), new ArrayList<WDLWorker.Data>());
        } else {
            if (validationRate != 0d) {
                this.trainingData = new MemoryLimitedList<WDLWorker.Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * (1 - validationRate)),
                        new ArrayList<WDLWorker.Data>());
                this.validationData = new MemoryLimitedList<WDLWorker.Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * validationRate),
                        new ArrayList<WDLWorker.Data>());
            } else {
                this.trainingData = new MemoryLimitedList<WDLWorker.Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction), new ArrayList<WDLWorker.Data>());
            }
        }

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        // numerical + categorical = # of all input
        this.numInputs = inputOutputIndex[0];
        this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
        // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it is
        // 1, with index of 0,1,2,3 denotes different classes
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null
                && !"".equals(modelConfig.getValidationDataSetRawPath()));

        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();

        //build multi-task nn model:
        this.validParams = this.modelConfig.getTrain().getParams();
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        List<String> hiddenActiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        int taskNumber = 0;
        for (ColumnConfig cConfig : this.columnConfigList) {
            ColumnConfig.ColumnFlag flag = ColumnConfig.ColumnFlag.Target;
            if (cConfig.getColumnFlag().equals(flag)) {
                taskNumber++;
            }
        }
        // todo:check if MTNN need regression function
        double l2reg = NumberUtils.toDouble(this.validParams.get(CommonConstants.WDL_L2_REG).toString(), 0);
        this.mtnn = new MultiTaskNN(numInputs, hiddenNodes, hiddenActiFuncs, taskNumber, l2reg);

    }

    @Override
    public MTNNParams doCompute(WorkerContext<MTNNParams, MTNNParams> context) {
        if (context.isFirstIteration()){
            return new MTNNParams();
        }

        this.mtnn.updateWeights(context.getLastMasterResult());
        MTNNParallelGradient parallelGradient = new MTNNParallelGradient( this.workerThreadCount,this.mtnn,
                this.trainingData,this.validationData,this.completionService);
        MTNNParams mtnnParams = parallelGradient.doCompute();

        mtnnParams.setSerializationType(SerializationType.GRADIENTS);
        this.mtnn.setSerializationType(SerializationType.GRADIENTS);
        return mtnnParams;
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue, WorkerContext<MTNNParams, MTNNParams> context) {

    }
}
