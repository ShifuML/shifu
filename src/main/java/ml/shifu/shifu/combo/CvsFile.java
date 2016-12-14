package ml.shifu.shifu.combo;

import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

/**
 * Created by zhanhu on 12/10/16.
 */
public class CvsFile implements Iterable<Map<String, String>>{
    private static Logger LOG = LoggerFactory.getLogger(CvsFile.class);

    private String filePath;
    private String delimiter;

    public CvsFile(String filePath, String delimiter) {
        this.filePath = filePath;
        this.delimiter = delimiter;
    }

    @Override
    public Iterator<Map<String, String>> iterator() {
        return new CvsFileIterator(filePath, delimiter);
    }

    public class CvsFileIterator implements Iterator<Map<String, String>> {
        private String nextLine;
        private BufferedReader reader;
        private boolean isFinished = false;
        private String[] header;
        private String delimiter;

        public CvsFileIterator(String filePath, String delimiter) {
            LOG.info("Create csv file {}, with delimiter {}", filePath, delimiter);
            this.delimiter = delimiter;
            try {
                open(filePath);
            } catch (Exception e) {
                LOG.error("Fail to open file - {}.", filePath,  e);
            }
        }

        private void open(String filePath) throws IOException {
            this.reader = new BufferedReader(new FileReader(filePath));
            String headerLine = reader.readLine();
            this.header = StringUtils.split(headerLine, delimiter);
        }

        @Override
        public boolean hasNext() {
            if ( nextLine != null ) {
                return true;
            } else if ( isFinished ) {
                return false;
            } else {
                try {
                    nextLine = reader.readLine();
                } catch (IOException e) {
                    nextLine = null;
                }

                if ( nextLine == null ) {
                    isFinished = true;
                    IOUtils.closeQuietly(reader);
                    return false;
                }

                return true;
            }
        }

        @Override
        public Map<String, String> next() {
            if ( ! hasNext() ) {
                return null;
            }

            String[] vars = StringUtils.split(nextLine, delimiter);
            Map<String, String> varMap = new HashedMap();
            for ( int i = 0; i < this.header.length; i ++ ) {
                varMap.put(this.header[i], vars[i]);
            }

            nextLine = null;

            return varMap;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Remove is not supported.");
        }
    }
}
