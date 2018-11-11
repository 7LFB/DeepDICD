def build_centroid_based_losses_v2(self):
    source_softmax = self.slabel
    target_softmax = tf.nn.softmax(self.tlabel_pred, axis=1)
    ones = tf.ones_like(self.sfc8)
    current_source_count = tf.unsorted_segment_sum(
        ones, source_label, self.params.num_classes)
    current_target_count = tf.reduce_sum(target_softmax, keep_dims=True)

    current_positive_source_count = tf.maximum(
        current_source_count, tf.ones_like(current_source_count))

    current_source_centroid = tf.divide(tf.unsorted_segment_sum(
        data=self.sfc8, segment_ids=source_label, num_segments=self.params.num_classes), current_positive_source_count)
    current_target_centroid = tf.divide(
        tf.matmul(target_softmax, self.tfc8, False, True), current_target_count)

    decay = tf.constant(0.3)

    source_moving_centroid = (
        decay) * current_source_centroid + (1. - decay) * self.source_moving_centroid
    target_moving_centroid = (
        decay) * current_target_centroid + (1. - decay) * self.target_moving_centroid

    self.class_wise_adaptation_loss = self.build_class_wise_adaptation_losses(
        source_moving_centroid, target_moving_centroid)
    self.sintra_loss = self.build_batch_intra_losses(
        source_moving_centroid, self.sfc8, self.slabel)
    self.tintra_loss = self.build_batch_intra_losses(
        target_moving_centroid, self.tfc8, tf.argmax(self.tlabel_pred, 1))
    self.sinter_loss = self.build_batch_inter_losses(
        source_moving_centroid, self.sfc8, self.slabel)
    self.tinter_loss = self.build_batch_inter_losses(
        target_moving_centroid, self.tfc8, tf.argmax(self.tlabel_pred, 1))

    update_src = self.source_moving_centroid.assign(source_moving_centroid)
    update_tar = self.target_moving_centroid.assign(target_moving_centroid)

    return update_src, update_tar
