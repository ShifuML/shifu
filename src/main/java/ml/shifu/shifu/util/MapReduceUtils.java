package ml.shifu.shifu.util;

import com.google.common.base.Splitter;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by zhanhu on 4/9/18.
 */
public class MapReduceUtils {

    private static final Logger LOG = LoggerFactory.getLogger(MapReduceUtils.class);

    /**
     * Since we allow user to define the delimiter for Shifu output, value like '\u0007', '\u0010' will be
     * legal separator. But in Hadoop job, those chars are illegal in job.xml. Before passing to MR jobs, the delimiter
     * is base64-encoded. This function is try to decode the delimiter and build Splitter for MR jobs.
     * @param delimiter - delimiter in context or properties
     * @return - Splitter for MR jobs.
     */
    public static Splitter generateShifuOutputSplitter(String delimiter) {
        try {
            delimiter = (StringUtils.isNotBlank(delimiter)
                    ? Base64Utils.base64Decode(delimiter) : Constants.DEFAULT_DELIMITER);
        } catch (Exception e) {
            delimiter = Constants.DEFAULT_DELIMITER;
        }
        LOG.info("The delimiter of normalization data is - {}", delimiter);
        return Splitter.on(delimiter).trimResults();
    }
}
