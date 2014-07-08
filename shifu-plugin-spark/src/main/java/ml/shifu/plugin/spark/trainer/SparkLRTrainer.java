package ml.shifu.plugin.spark.trainer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.trainer.StaticFunctions.ParsePoint;
import ml.shifu.plugin.spark.trainer.StaticFunctions.ParseVector;
import ml.shifu.plugin.spark.trainer.StaticFunctions.TargetVector;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.primitives.Ints;


public class SparkLRTrainer implements Trainer {

    private static final String INPUTDATASET = "pathNormalizedData";
    private static final String EVALDATASET = "evalDataSet";
    private static final String SPLITTER = "splitter";
    private static final String TRAINERID = "trainerID";
    private static Logger log = LoggerFactory.getLogger(SparkLRTrainer.class);
    private static final DecimalFormat df = new DecimalFormat("0.000000");
    private LogisticRegressionModel lrModel;
    private Params rawParams;
    private MiningSchema schema;

    public void train(PMMLDataSet dataSet, Params rawParams) throws Exception {
        train(dataSet.getMiningSchema(), rawParams);
    }

    public void train(MiningSchema schema, Params rawParams) throws Exception {
        this.rawParams = rawParams;
        this.schema = schema;
        SparkLRParams lrParams = parseModelParams(rawParams);
        String trainerID = rawParams.get(TRAINERID).toString();
        // prepare data set
        RDD<LabeledPoint> points = convertRDDData(getTargetFieldID(), getActiveFieldID());
        // train the data
        lrModel = LogisticRegressionWithSGD.train(points,
                lrParams.getIterations(), lrParams.getStepSize());
        lrModel.clearThreshold();
        // evaluate and calculate errors
        log.info("  Trainer-" + trainerID + "\n Train Error: "
                + df.format(getTestSetError()) );
        log.info("Trainer #" + trainerID + " is Finished!");
    }

    private List<Integer> getFieldIDViaUsageType(FieldUsageType usageType) {
        List<MiningField> miningFields = schema.getMiningFields();
        List<Integer> idList = new ArrayList<Integer>();
        for (int i = 0; i < miningFields.size(); i++) {
            if (miningFields.get(i).getUsageType().equals(usageType))
                idList.add(i);
        }
        return idList;
    }

    private int[] getActiveFieldID() {
        return Ints.toArray(getFieldIDViaUsageType(FieldUsageType.ACTIVE));
    }

    private int getTargetFieldID() {
        List<Integer> targetID = getFieldIDViaUsageType(FieldUsageType.TARGET);
        return targetID.get(0);
    }

    private RDD<LabeledPoint> convertRDDData(int targetID, int[] activeIndexSet) {
        JavaRDD<String> lines = SparkUtility.getSc().textFile(
                rawParams.get(INPUTDATASET).toString());
        RDD<LabeledPoint> datas = lines
                .map(new ParsePoint(targetID, activeIndexSet, rawParams.get(
                        SPLITTER).toString())).cache().rdd();
        return datas;
    }

    private Double calculateMSE() {
        final int[] activeIndexSet = getActiveFieldID();
        JavaRDD<String> lines = SparkUtility.getSc().textFile(
                rawParams.get(EVALDATASET).toString());
        JavaRDD<Vector> evalVectors = lines.map(
                new ParseVector(activeIndexSet, rawParams.get(SPLITTER)
                        .toString())).cache();
        List<Double> idealDatas = lines
                .map(new TargetVector(getTargetFieldID(), rawParams.get(
                        SPLITTER).toString())).cache().collect();
      //TODO migrate to Spark Function
        double mseError = 0;
        int numRecords = idealDatas.size();
        List<Double> evalList = lrModel.predict(evalVectors).cache().collect();
        for (int i = 0; i < numRecords; i++)
            mseError += Math.pow(evalList.get(i) - idealDatas.get(i), 2.0);

        return mseError / numRecords;
    }

    private SparkLRParams parseModelParams(Params rawParams) throws Exception {
        ObjectMapper jsonMapper = new ObjectMapper();
        String jsonString = jsonMapper.writeValueAsString(rawParams);
        return jsonMapper.readValue(jsonString, SparkLRParams.class);
    }
   
    private double getTestSetError() {
        return calculateMSE();
    }

//    private void saveLR(String path) throws IOException {
//
//        // EncogDirectoryPersistence.saveObject(new File(path), network);
//    }

}
