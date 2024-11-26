package ml.shifu.shifu.fs;

import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.apache.hadoop.fs.Path;
import org.apache.logging.log4j.util.Strings;
import parquet.example.data.Group;
import parquet.example.data.simple.SimpleGroup;
import parquet.hadoop.ParquetReader;
import parquet.hadoop.example.GroupReadSupport;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParquetInputStream extends InputStream {

    private final ParquetReader<Group> reader;
    private static final String FIELD_DELIMITER = Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER,
            Constants.DEFAULT_DELIMITER);
    private static final Integer ALL_DONE = -1;
    private ByteArrayInputStream buffer = new ByteArrayInputStream(new byte[0]);
    private final Charset charset;

    public ParquetInputStream(Path filePath, String charset) throws IOException {
        this.reader = ParquetReader.builder(new GroupReadSupport(), filePath).build();
        this.charset = Charset.forName(charset);
    }

    @Override
    public int read() throws IOException {
        int readOne = buffer.read();
        while (readOne == ALL_DONE) {
            Group record = this.reader.read();
            if(record == null) {
                return ALL_DONE;
            } else {
                this.buffer = buildScanBuffer(record);
            }
            readOne = this.buffer.read();
        }
        return readOne;
    }

    private ByteArrayInputStream buildScanBuffer(Group group) {
        String record = simpleGroupToString((SimpleGroup) group) + Strings.LINE_SEPARATOR;
        return new ByteArrayInputStream(record.getBytes(this.charset));
    }

    private String simpleGroupToString(SimpleGroup group) {
        return IntStream.range(0, group.getType().getFieldCount())
                .mapToObj(i -> {
                    int fieldRepetitionCount = group.getFieldRepetitionCount(i);
                    if(fieldRepetitionCount > 0) {
                        return group.getValueToString(i, 0);
                    }
                    return Strings.EMPTY;
                })
                .collect(Collectors.joining(FIELD_DELIMITER));
    }

    @Override
    public void close() throws IOException {
        this.reader.close();
    }
}