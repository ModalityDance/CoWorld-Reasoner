"""Infrastructure failures are classified separately from completed reviews."""
VERIFIER_TRANSPORT_FAILURES = frozenset({
    'error', 'failed', 'timeout', 'video_verifier_failed',
    'video_verifier_not_configured',
})


def verifier_transport_failure(records):
    for record in records:
        status = (record.get('result') or {}).get('status')
        if status in VERIFIER_TRANSPORT_FAILURES:
            return f'verifier_transport_failed:{status}'
    return None
