package ml.shifu.plugin.spark.norm;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

public class SparkOutputFileNameFilter implements PathFilter {

    public boolean accept(Path path) {
        String name = path.getName();
        if (name.startsWith("part-") && !name.endsWith(".crc"))
            return true;
        else
            return false;
    }
}
