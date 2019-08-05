package ml.shifu.shifu.util;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ModelConfig;
import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class MultiClsTagPredictor {

    private static Logger LOG = LoggerFactory.getLogger(MultiClsTagPredictor.class);
    @SuppressWarnings("unused")
    private ModelConfig modelConfig;
    private boolean isOneVsAll;
    private List<String> tags;

    public MultiClsTagPredictor(ModelConfig modelConfig) {
        assert modelConfig.isClassification();

        this.modelConfig = modelConfig;
        this.isOneVsAll = modelConfig.getTrain().isOneVsAll();
        this.tags = modelConfig.getTags();
    }

    public PredictResult predictTag(CaseScoreResult sc) {
        PredictResult ret = null;
        if (isOneVsAll) {
            ret = predictMultiClsTag(tags, sc.getScores());
        } else {
            int modelCnt = sc.getScores().size() / tags.size();

            List<PredictResult> results = new ArrayList<PredictResult>(modelCnt);
            for (int i = 0; i < modelCnt; i++) {
                PredictResult result = predictMultiClsTag(tags,
                        sc.getScores().subList(i * tags.size(), (i + 1) * tags.size()));
                if (result != null) {
                    results.add(result);
                }
            }

            ret = voteFinalResult(tags, results);
        }

        return ret;
    }

    private PredictResult voteFinalResult(List<String> tags, List<PredictResult> results) {
        int[] votes = new int[tags.size()];
        for (PredictResult result : results) {
            int pos = tags.indexOf(result.getTag());
            votes[pos] = votes[pos] + 1;
        }

        if (isEqualVotes(votes)) {
            return findBestPredictResult(results);
        } else {
            int maxVoteCnt = votes[0];
            int pos = 0;
            for (int i = 1; i < votes.length; i++) {
                if (votes[i] > maxVoteCnt) {
                    maxVoteCnt = votes[i];
                    pos = i;
                }
            }

            PredictResult ret = null;
            for (PredictResult temp : results) {
                if (temp.getTag().equals(tags.get(pos))
                        && (ret == null || temp.getConfidence() > ret.getConfidence())) {
                    ret = temp;
                }
            }
            return ret;
        }
    }

    private PredictResult findBestPredictResult(List<PredictResult> results) {
        PredictResult ret = null;
        for (PredictResult temp : results) {
            if (ret == null || temp.getConfidence() > ret.getConfidence()) {
                ret = temp;
            }
        }
        return ret;
    }

    private boolean isEqualVotes(int[] votes) {
        int vote = votes[0];
        for (int v : votes) {
            if (v != vote) {
                return false;
            }
        }
        return true;
    }

    private PredictResult predictMultiClsTag(List<String> tags, List<Double> scores) {
        assert (CollectionUtils.isNotEmpty(tags) && CollectionUtils.isNotEmpty(scores) && tags.size() == scores.size());
        int pos = 0;
        double maxScore = scores.get(0);
        for (int i = 1; i < scores.size(); i++) {
            if (scores.get(i) > maxScore) {
                maxScore = scores.get(i);
                pos = i;
            }
        }
        return new PredictResult(tags.get(pos), maxScore);
    }

    public static class PredictResult {
        private String tag;
        private double confidence;

        public PredictResult(String tag, double confidence) {
            this.tag = tag;
            this.confidence = confidence;
        }

        public String getTag() {
            return tag;
        }

        public void setTag(String tag) {
            this.tag = tag;
        }

        public double getConfidence() {
            return confidence;
        }

        public void setConfidence(double confidence) {
            this.confidence = confidence;
        }
    }
}
