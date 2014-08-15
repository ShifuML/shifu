/*
 * This is the user defined function which is called by Spark "map" on each record in the JavaRDD.
 * This map should be applied on the raw JavaRDD containing strings of raw rows from the input data file.
 * This class is initialized with an BroadcastVariables object as the single broadcast variable.
 * Normalization takes a single row of data and applies transformations defined in the PMML object by 
 * calling the transform() function of the TransformationExecutor object.
 * The resulting normalized row is again converted to a single string which is output.
 */

package ml.shifu.plugin.spark.norm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import com.google.common.base.Joiner;

public class Normalize implements Function<String, String> {
    // private BroadcastVariables bVar;
    // TODO: get this from bVar
    private Broadcast<BroadcastVariables> broadVar;
    private String pattern;

    public Normalize(Broadcast<BroadcastVariables> bVar) {
        // this.bVar= bVar.value();
        this.broadVar = bVar;
        // this.bvar= bVar.value();
        // create pattern to be used for printing out normalized floats/ doubles
        // based on precision
        pattern = "%." + bVar.value().getPrecision() + "f";
    }

    @Override
    public String call(String input) {

        // List<Object> parsedInput= CombinedUtils.getParsedObjects(input,
        // bvar.getDelimiter());
        // Map<String, Object> rawDataMap=
        // CombinedUtils.createDataMap(bvar.getDataFields(), parsedInput);
        Map<String, Object> rawDataMap = CombinedUtils.createDataMap(broadVar
                .value().getDataFields(), input, broadVar.value()
                .getDelimiter());
        List<Object> result = broadVar.value().getExec()
                .transform(broadVar.value().getTargetFields(), rawDataMap);
        result.addAll(broadVar.value().getExec()
                .transform(broadVar.value().getActiveFields(), rawDataMap));
        List<String> resultStr = new ArrayList<String>();

        // if object is float/ double apply pattern before joining
        for (Object r : result) {
            if (r instanceof Float || r instanceof Double) {
                resultStr.add(String.format(pattern, r));
            } else
                resultStr.add(r.toString());
        }

        return Joiner.on(broadVar.value().getDelimiter()).join(resultStr);
    }
}
