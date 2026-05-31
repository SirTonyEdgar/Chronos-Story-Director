import React from 'react';
import SharedEditor from './SharedEditor';
import { BookMarked } from 'lucide-react';

export default function ReferenceTab({ profile }) {
  return (
    <SharedEditor
      profile={profile}
      category="Reference"
      icon={<BookMarked />}
      color="#06b6d4"
      description={`📚 Style references teach the AI how to write (prose rhythm, voice, tone) — prefix the title with [Style] to mark one.\n\nWorld references teach it how things work — period detail, real-world mechanics, domain knowledge.\n\nTo preserve minor characters across scenes, create a World reference titled 'Recurring Voices' and list each one with a name and brief description.`}
      placeholder="e.g. Paste a McCarthy paragraph for prose rhythm reference, or describe how 1930s naval intelligence actually communicated..."
    />
  );
}