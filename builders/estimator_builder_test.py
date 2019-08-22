from builders import estimator_builder
from protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow as tf


class EstimatorBuilderTest(tf.test.TestCase):
    def test_estimator_config(self):
        proto_txt = """
        experiment_dir: "/home/abc"
        experiment_name : "abc"
        """
        proto_msg = pipeline_pb2.ExperimentConfig()
        text_format.Merge(proto_txt,
                          proto_msg)

        estimator_config = estimator_builder.build_estimator_config(proto_msg)
        expected_config = tf.estimator.RunConfig(
            model_dir='/home/abc',
            tf_random_seed=0,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            keep_checkpoint_max=20,
            log_step_count_steps=100
        )
        self.assertEqual(estimator_config.__dict__,
                         expected_config.__dict__)


if __name__ == "__main__":
    tf.app.main()





