package ml.shifu.plugin.spark.trainer;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkUtility {
    public static SparkConf sparkConf = new SparkConf().setMaster("local")
            .setAppName("PMMLSparkLR").setSparkHome("localhost:4040");

    public static JavaSparkContext sc = new JavaSparkContext(sparkConf);


    public static SparkConf getSparkConf() {
        return sparkConf;
    }

    public static JavaSparkContext getSc() {
        return sc;
    }


}
