from django.db import models
from django.utils import timezone

class TranslationSession(models.Model):
    """Tracks a single continuous usage session of the Mudrā camera interface."""
    session_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    notes = models.CharField(max_length=255, default="Live Stream Session", blank=True)

    class Meta:
        db_table = 'translation_sessions'
        ordering = ['-start_time']

    def __str__(self):
        return f"Session #{self.session_id} - {self.start_time.strftime('%Y-%m-%d %H:%M')}"


class TranslationLog(models.Model):
    """Logs every individual high-confidence gesture classification predicted by the LSTM network."""
    log_id = models.BigAutoField(primary_key=True)
    session = models.ForeignKey(TranslationSession, on_delete=models.CASCADE, related_name='logs')
    predicted_word = models.CharField(max_length=50)
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'translation_logs'
        ordering = ['timestamp']

    def __str__(self):
        return f"[{self.predicted_word}] Conf: {self.confidence_score:.2f} at {self.timestamp.strftime('%H:%M:%S')}"