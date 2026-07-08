interface Column<T> {
  header: string
  render: (row: T) => React.ReactNode
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  keyFn: (row: T) => string
  emptyMessage?: string
}

export function DataTable<T>({ columns, rows, keyFn, emptyMessage }: DataTableProps<T>) {
  if (rows.length === 0) {
    return <p className="caption">{emptyMessage ?? 'Nothing here yet.'}</p>
  }
  return (
    <table>
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col.header}>{col.header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={keyFn(row)}>
            {columns.map((col) => (
              <td key={col.header}>{col.render(row)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
