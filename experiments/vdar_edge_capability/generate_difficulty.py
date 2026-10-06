# Resumable 134-task difficulty generation entry point.
if __package__:
    from .difficulty_batch import main, load_tasks
else:
    from difficulty_batch import main, load_tasks


if __name__ == '__main__':
    main()
