interface Column<T> {
  header: string
  render: (row: T) => React.ReactNode
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  keyFn: (row: T) => string
  emptyMessage?: string
  /** When set, caps the visible height to roughly this many body rows and scrolls the rest vertically instead of growing the page. */
  maxVisibleRows?: number
}

const HEADER_ROW_HEIGHT = 42
const BODY_ROW_HEIGHT = 42

export function DataTable<T>({ columns, rows, keyFn, emptyMessage, maxVisibleRows }: DataTableProps<T>) {
  if (rows.length === 0) {
    return <p className="caption">{emptyMessage ?? 'Nothing here yet.'}</p>
  }
  return (
    <div
      className={maxVisibleRows ? 'data-table-scroll' : undefined}
      style={{
        overflowX: 'auto',
        ...(maxVisibleRows
          ? { maxHeight: HEADER_ROW_HEIGHT + maxVisibleRows * BODY_ROW_HEIGHT, overflowY: 'auto' }
          : {}),
      }}
    >
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
    </div>
  )
}
