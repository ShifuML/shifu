package ml.shifu.plugin.spark.stats;

import java.util.List;

import org.apache.spark.Accumulable;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.DataField;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;

import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;
import ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator;

/*
 * Implementation of SparkStatsCalculator for Binomial Stats.
 */
public class BinomialStatsCalculator implements SparkStatsCalculator {

    public ModelStats calculate(JavaSparkContext jsc, JavaRDD<String> data, PMML pmml, Params bindingParams) {
        List<DataField> dataFields= pmml.getDataDictionary().getDataFields();
        int targetFieldNum = PMMLUtils.getTargetFieldNumByName(pmml.getDataDictionary(), (String) bindingParams.get("targetFieldName"));
        bindingParams.put("targetFieldNum", targetFieldNum);
        Accumulable<ColumnStateArray, String> accum= jsc.accumulable(new BinomialColumnStateArray(dataFields, bindingParams), new SparkAccumulableWrapper());
        data.foreach(new AccumulatorApplicator(accum));
        ColumnStateArray colStateArray= accum.value();
        return colStateArray.getModelStats();
    }

}
